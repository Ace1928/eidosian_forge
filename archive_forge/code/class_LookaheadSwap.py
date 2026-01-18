import collections
import copy
import logging
import math
from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout
from qiskit.dagcircuit import DAGOpNode
from qiskit.transpiler.target import Target
from qiskit.transpiler.passes.layout import disjoint_utils
class LookaheadSwap(TransformationPass):
    """Map input circuit onto a backend topology via insertion of SWAPs.

    Implementation of Sven Jandura's swap mapper submission for the 2018 Qiskit
    Developer Challenge, adapted to integrate into the transpiler architecture.

    The role of the swapper pass is to modify the starting circuit to be compatible
    with the target device's topology (the set of two-qubit gates available on the
    hardware.) To do this, the pass will insert SWAP gates to relocate the virtual
    qubits for each upcoming gate onto a set of coupled physical qubits. However, as
    SWAP gates are particularly lossy, the goal is to accomplish this remapping while
    introducing the fewest possible additional SWAPs.

    This algorithm searches through the available combinations of SWAP gates by means
    of a narrowed best first/beam search, described as follows:

    - Start with a layout of virtual qubits onto physical qubits.
    - Find any gates in the input circuit which can be performed with the current
      layout and mark them as mapped.
    - For all possible SWAP gates, calculate the layout that would result from their
      application and rank them according to the distance of the resulting layout
      over upcoming gates (see _calc_layout_distance.)
    - For the four (search_width) highest-ranking SWAPs, repeat the above process on
      the layout that would be generated if they were applied.
    - Repeat this process down to a depth of four (search_depth) SWAPs away from the
      initial layout, for a total of 256 (search_width^search_depth) prospective
      layouts.
    - Choose the layout which maximizes the number of two-qubit which could be
      performed. Add its mapped gates, including the SWAPs generated, to the
      output circuit.
    - Repeat the above until all gates from the initial circuit are mapped.

    For more details on the algorithm, see Sven's blog post:
    https://medium.com/qiskit/improving-a-quantum-compiler-48410d7a7084
    """

    def __init__(self, coupling_map, search_depth=4, search_width=4, fake_run=False):
        """LookaheadSwap initializer.

        Args:
            coupling_map (Union[CouplingMap, Target]): CouplingMap of the target backend.
            search_depth (int): lookahead tree depth when ranking best SWAP options.
            search_width (int): lookahead tree width when ranking best SWAP options.
            fake_run (bool): if true, it will only pretend to do routing, i.e., no
                swap is effectively added.
        """
        super().__init__()
        if isinstance(coupling_map, Target):
            self.target = coupling_map
            self.coupling_map = self.target.build_coupling_map()
        else:
            self.target = None
            self.coupling_map = coupling_map
        self.search_depth = search_depth
        self.search_width = search_width
        self.fake_run = fake_run

    def run(self, dag):
        """Run the LookaheadSwap pass on `dag`.

        Args:
            dag (DAGCircuit): the directed acyclic graph to be mapped
        Returns:
            DAGCircuit: A dag mapped to be compatible with the coupling_map in
                the property_set.
        Raises:
            TranspilerError: if the coupling map or the layout are not
            compatible with the DAG, or if the coupling_map=None
        """
        if self.coupling_map is None:
            raise TranspilerError('LookaheadSwap cannot run with coupling_map=None')
        if len(dag.qregs) != 1 or dag.qregs.get('q', None) is None:
            raise TranspilerError('Lookahead swap runs on physical circuits only')
        number_of_available_qubits = len(self.coupling_map.physical_qubits)
        if len(dag.qubits) > number_of_available_qubits:
            raise TranspilerError(f'The number of DAG qubits ({len(dag.qubits)}) is greater than the number of available device qubits ({number_of_available_qubits}).')
        disjoint_utils.require_layout_isolated_to_component(dag, self.coupling_map if self.target is None else self.target)
        register = dag.qregs['q']
        current_state = _SystemState(Layout.generate_trivial_layout(register), self.coupling_map, register)
        mapped_gates = []
        gates_remaining = list(dag.serial_layers())
        while gates_remaining:
            logger.debug('Top-level routing step: %d gates remaining.', len(gates_remaining))
            best_step = _search_forward_n_swaps(current_state, gates_remaining, self.search_depth, self.search_width)
            if best_step is None:
                raise TranspilerError('Lookahead failed to find a swap which mapped gates or improved layout score.')
            logger.debug('Found best step: mapped %d gates. Added swaps: %s.', len(best_step.gates_mapped), best_step.swaps_added)
            current_state = best_step.state
            gates_mapped = best_step.gates_mapped
            gates_remaining = best_step.gates_remaining
            mapped_gates.extend(gates_mapped)
        self.property_set['final_layout'] = current_state.layout
        if self.fake_run:
            return dag
        mapped_dag = dag.copy_empty_like()
        for node in mapped_gates:
            mapped_dag.apply_operation_back(node.op, node.qargs, node.cargs, check=False)
        return mapped_dag