import logging
from copy import deepcopy
import time
import rustworkx
from qiskit.circuit import SwitchCaseOp, ControlFlowOp, Clbit, ClassicalRegister
from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.circuit.controlflow import condition_resources, node_resources
from qiskit.converters import dag_to_circuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.target import Target
from qiskit.transpiler.passes.layout import disjoint_utils
from qiskit.dagcircuit import DAGCircuit
from qiskit.utils.parallel import CPU_COUNT
from qiskit._accelerate.sabre_swap import (
from qiskit._accelerate.nlayout import NLayout
class SabreSwap(TransformationPass):
    """Map input circuit onto a backend topology via insertion of SWAPs.

    Implementation of the SWAP-based heuristic search from the SABRE qubit
    mapping paper [1] (Algorithm 1). The heuristic aims to minimize the number
    of lossy SWAPs inserted and the depth of the circuit.

    This algorithm starts from an initial layout of virtual qubits onto physical
    qubits, and iterates over the circuit DAG until all gates are exhausted,
    inserting SWAPs along the way. It only considers 2-qubit gates as only those
    are germane for the mapping problem (it is assumed that 3+ qubit gates are
    already decomposed).

    In each iteration, it will first check if there are any gates in the
    ``front_layer`` that can be directly applied. If so, it will apply them and
    remove them from ``front_layer``, and replenish that layer with new gates
    if possible. Otherwise, it will try to search for SWAPs, insert the SWAPs,
    and update the mapping.

    The search for SWAPs is restricted, in the sense that we only consider
    physical qubits in the neighborhood of those qubits involved in
    ``front_layer``. These give rise to a ``swap_candidate_list`` which is
    scored according to some heuristic cost function. The best SWAP is
    implemented and ``current_layout`` updated.

    This transpiler pass adds onto the SABRE algorithm in that it will run
    multiple trials of the algorithm with different seeds. The best output,
    determined by the trial with the least amount of SWAPed inserted, will
    be selected from the random trials.

    **References:**

    [1] Li, Gushu, Yufei Ding, and Yuan Xie. "Tackling the qubit mapping problem
    for NISQ-era quantum devices." ASPLOS 2019.
    `arXiv:1809.02573 <https://arxiv.org/pdf/1809.02573.pdf>`_
    """

    def __init__(self, coupling_map, heuristic='basic', seed=None, fake_run=False, trials=None):
        """SabreSwap initializer.

        Args:
            coupling_map (Union[CouplingMap, Target]): CouplingMap of the target backend.
            heuristic (str): The type of heuristic to use when deciding best
                swap strategy ('basic' or 'lookahead' or 'decay').
            seed (int): random seed used to tie-break among candidate swaps.
            fake_run (bool): if true, it only pretend to do routing, i.e., no
                swap is effectively added.
            trials (int): The number of seed trials to run sabre with. These will
                be run in parallel (unless the PassManager is already running in
                parallel). If not specified this defaults to the number of physical
                CPUs on the local system. For reproducible results it is recommended
                that you set this explicitly, as the output will be deterministic for
                a fixed number of trials.

        Raises:
            TranspilerError: If the specified heuristic is not valid.

        Additional Information:

            The search space of possible SWAPs on physical qubits is explored
            by assigning a score to the layout that would result from each SWAP.
            The goodness of a layout is evaluated based on how viable it makes
            the remaining virtual gates that must be applied. A few heuristic
            cost functions are supported

            - 'basic':

            The sum of distances for corresponding physical qubits of
            interacting virtual qubits in the front_layer.

            .. math::

                H_{basic} = \\sum_{gate \\in F} D[\\pi(gate.q_1)][\\pi(gate.q2)]

            - 'lookahead':

            This is the sum of two costs: first is the same as the basic cost.
            Second is the basic cost but now evaluated for the
            extended set as well (i.e. :math:`|E|` number of upcoming successors to gates in
            front_layer F). This is weighted by some amount EXTENDED_SET_WEIGHT (W) to
            signify that upcoming gates are less important that the front_layer.

            .. math::

                H_{decay}=\\frac{1}{\\left|{F}\\right|}\\sum_{gate \\in F} D[\\pi(gate.q_1)][\\pi(gate.q2)]
                    + W*\\frac{1}{\\left|{E}\\right|} \\sum_{gate \\in E} D[\\pi(gate.q_1)][\\pi(gate.q2)]

            - 'decay':

            This is the same as 'lookahead', but the whole cost is multiplied by a
            decay factor. This increases the cost if the SWAP that generated the
            trial layout was recently used (i.e. it penalizes increase in depth).

            .. math::

                H_{decay} = max(decay(SWAP.q_1), decay(SWAP.q_2)) {
                    \\frac{1}{\\left|{F}\\right|} \\sum_{gate \\in F} D[\\pi(gate.q_1)][\\pi(gate.q2)]\\\\
                    + W *\\frac{1}{\\left|{E}\\right|} \\sum_{gate \\in E} D[\\pi(gate.q_1)][\\pi(gate.q2)]
                    }
        """
        super().__init__()
        if isinstance(coupling_map, Target):
            self.target = coupling_map
            self.coupling_map = self.target.build_coupling_map()
        else:
            self.coupling_map = coupling_map
            self.target = None
        if self.coupling_map is not None and (not self.coupling_map.is_symmetric):
            if isinstance(coupling_map, CouplingMap):
                self.coupling_map = deepcopy(self.coupling_map)
            self.coupling_map.make_symmetric()
        self._neighbor_table = None
        if self.coupling_map is not None:
            self._neighbor_table = NeighborTable(rustworkx.adjacency_matrix(self.coupling_map.graph))
        self.heuristic = heuristic
        self.seed = seed
        if trials is None:
            self.trials = CPU_COUNT
        else:
            self.trials = trials
        self.fake_run = fake_run
        self._qubit_indices = None
        self._clbit_indices = None
        self.dist_matrix = None

    def run(self, dag):
        """Run the SabreSwap pass on `dag`.

        Args:
            dag (DAGCircuit): the directed acyclic graph to be mapped.
        Returns:
            DAGCircuit: A dag mapped to be compatible with the coupling_map.
        Raises:
            TranspilerError: if the coupling map or the layout are not
            compatible with the DAG, or if the coupling_map=None
        """
        if self.coupling_map is None:
            raise TranspilerError('SabreSwap cannot run with coupling_map=None')
        if len(dag.qregs) != 1 or dag.qregs.get('q', None) is None:
            raise TranspilerError('Sabre swap runs on physical circuits only.')
        num_dag_qubits = len(dag.qubits)
        num_coupling_qubits = self.coupling_map.size()
        if num_dag_qubits < num_coupling_qubits:
            raise TranspilerError(f'Fewer qubits in the circuit ({num_dag_qubits}) than the coupling map ({num_coupling_qubits}). Have you run a layout pass and then expanded your DAG with ancillas? See `FullAncillaAllocation`, `EnlargeWithAncilla` and `ApplyLayout`.')
        if num_dag_qubits > num_coupling_qubits:
            raise TranspilerError(f'More qubits in the circuit ({num_dag_qubits}) than available in the coupling map ({num_coupling_qubits}). This circuit cannot be routed to this device.')
        if self.heuristic == 'basic':
            heuristic = Heuristic.Basic
        elif self.heuristic == 'lookahead':
            heuristic = Heuristic.Lookahead
        elif self.heuristic == 'decay':
            heuristic = Heuristic.Decay
        else:
            raise TranspilerError('Heuristic %s not recognized.' % self.heuristic)
        disjoint_utils.require_layout_isolated_to_component(dag, self.coupling_map if self.target is None else self.target)
        self.dist_matrix = self.coupling_map.distance_matrix
        canonical_register = dag.qregs['q']
        current_layout = Layout.generate_trivial_layout(canonical_register)
        self._qubit_indices = {bit: idx for idx, bit in enumerate(canonical_register)}
        layout_mapping = {self._qubit_indices[k]: v for k, v in current_layout.get_virtual_bits().items()}
        initial_layout = NLayout(layout_mapping, len(dag.qubits), self.coupling_map.size())
        sabre_dag, circuit_to_dag_dict = _build_sabre_dag(dag, self.coupling_map.size(), self._qubit_indices)
        sabre_start = time.perf_counter()
        *sabre_result, final_permutation = build_swap_map(len(dag.qubits), sabre_dag, self._neighbor_table, self.dist_matrix, heuristic, initial_layout, self.trials, self.seed)
        sabre_stop = time.perf_counter()
        logging.debug('Sabre swap algorithm execution complete in: %s', sabre_stop - sabre_start)
        self.property_set['final_layout'] = Layout(dict(zip(dag.qubits, final_permutation)))
        if self.fake_run:
            return dag
        return _apply_sabre_result(dag.copy_empty_like(), dag, sabre_result, initial_layout, dag.qubits, circuit_to_dag_dict)