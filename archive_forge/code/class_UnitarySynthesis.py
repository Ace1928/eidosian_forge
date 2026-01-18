from __future__ import annotations
from math import pi, inf, isclose
from typing import Any
from copy import deepcopy
from itertools import product
from functools import partial
import numpy as np
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler import CouplingMap, Target
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.synthesis.one_qubit import one_qubit_decompose
from qiskit.synthesis.two_qubit.xx_decompose import XXDecomposer, XXEmbodiments
from qiskit.synthesis.two_qubit.two_qubit_decompose import (
from qiskit.quantum_info import Operator
from qiskit.circuit import ControlFlowOp, Gate, Parameter
from qiskit.circuit.library.standard_gates import (
from qiskit.transpiler.passes.synthesis import plugin
from qiskit.transpiler.passes.optimization.optimize_1q_decomposition import (
from qiskit.providers.models import BackendProperties
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
from qiskit.exceptions import QiskitError
class UnitarySynthesis(TransformationPass):
    """Synthesize gates according to their basis gates."""

    def __init__(self, basis_gates: list[str]=None, approximation_degree: float | None=1.0, coupling_map: CouplingMap=None, backend_props: BackendProperties=None, pulse_optimize: bool | None=None, natural_direction: bool | None=None, synth_gates: list[str] | None=None, method: str='default', min_qubits: int=None, plugin_config: dict=None, target: Target=None):
        """Synthesize unitaries over some basis gates.

        This pass can approximate 2-qubit unitaries given some
        gate fidelities (either via ``backend_props`` or ``target``).
        More approximation can be forced by setting a heuristic dial
        ``approximation_degree``.

        Args:
            basis_gates (list[str]): List of gate names to target. If this is
                not specified the ``target`` argument must be used. If both this
                and the ``target`` are specified the value of ``target`` will
                be used and this will be ignored.
            approximation_degree (float): heuristic dial used for circuit approximation
                (1.0=no approximation, 0.0=maximal approximation). Approximation can
                make the synthesized circuit cheaper at the cost of straying from
                the original unitary. If None, approximation is done based on gate fidelities.
            coupling_map (CouplingMap): the coupling map of the backend
                in case synthesis is done on a physical circuit. The
                directionality of the coupling_map will be taken into
                account if ``pulse_optimize`` is ``True``/``None`` and ``natural_direction``
                is ``True``/``None``.
            backend_props (BackendProperties): Properties of a backend to
                synthesize for (e.g. gate fidelities).
            pulse_optimize (bool): Whether to optimize pulses during
                synthesis. A value of ``None`` will attempt it but fall
                back if it does not succeed. A value of ``True`` will raise
                an error if pulse-optimized synthesis does not succeed.
            natural_direction (bool): Whether to apply synthesis considering
                directionality of 2-qubit gates. Only applies when
                ``pulse_optimize`` is ``True`` or ``None``. The natural direction is
                determined by first checking to see whether the
                coupling map is unidirectional.  If there is no
                coupling map or the coupling map is bidirectional,
                the gate direction with the shorter
                duration from the backend properties will be used. If
                set to True, and a natural direction can not be
                determined, raises :class:`.TranspilerError`. If set to None, no
                exception will be raised if a natural direction can
                not be determined.
            synth_gates (list[str]): List of gates to synthesize. If None and
                ``pulse_optimize`` is False or None, default to
                ``['unitary']``. If ``None`` and ``pulse_optimize == True``,
                default to ``['unitary', 'swap']``
            method (str): The unitary synthesis method plugin to use.
            min_qubits: The minimum number of qubits in the unitary to synthesize. If this is set
                and the unitary is less than the specified number of qubits it will not be
                synthesized.
            plugin_config: Optional extra configuration arguments (as a ``dict``)
                which are passed directly to the specified unitary synthesis
                plugin. By default, this will have no effect as the default
                plugin has no extra arguments. Refer to the documentation of
                your unitary synthesis plugin on how to use this.
            target: The optional :class:`~.Target` for the target device the pass
                is compiling for. If specified this will supersede the values
                set for ``basis_gates``, ``coupling_map``, and ``backend_props``.

        Raises:
            TranspilerError: if ``method`` was specified but is not found in the
                installed plugins list. The list of installed plugins can be queried with
                :func:`~qiskit.transpiler.passes.synthesis.plugin.unitary_synthesis_plugin_names`
        """
        super().__init__()
        self._basis_gates = set(basis_gates or ())
        self._approximation_degree = approximation_degree
        self._min_qubits = min_qubits
        self.method = method
        self.plugins = None
        if method != 'default':
            self.plugins = plugin.UnitarySynthesisPluginManager()
        self._coupling_map = coupling_map
        self._backend_props = backend_props
        self._pulse_optimize = pulse_optimize
        self._natural_direction = natural_direction
        self._plugin_config = plugin_config
        self._target = target
        if target is not None:
            self._coupling_map = self._target.build_coupling_map()
        if synth_gates:
            self._synth_gates = synth_gates
        elif pulse_optimize:
            self._synth_gates = ['unitary', 'swap']
        else:
            self._synth_gates = ['unitary']
        self._synth_gates = set(self._synth_gates) - self._basis_gates
        if self.method != 'default' and self.method not in self.plugins.ext_plugins:
            raise TranspilerError(f"Specified method '{self.method}' not found in plugin list")

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the UnitarySynthesis pass on ``dag``.

        Args:
            dag: input dag.

        Returns:
            Output dag with UnitaryGates synthesized to target basis.
        """
        if not set(self._synth_gates).intersection(dag.count_ops()):
            return dag
        if self.plugins:
            plugin_method = self.plugins.ext_plugins[self.method].obj
        else:
            plugin_method = DefaultUnitarySynthesis()
        plugin_kwargs: dict[str, Any] = {'config': self._plugin_config}
        _gate_lengths = _gate_errors = None
        _gate_lengths_by_qubit = _gate_errors_by_qubit = None
        if self.method == 'default':
            default_method = plugin_method
            default_kwargs = plugin_kwargs
            method_list = [(plugin_method, plugin_kwargs)]
        else:
            default_method = self.plugins.ext_plugins['default'].obj
            default_kwargs = {}
            method_list = [(plugin_method, plugin_kwargs), (default_method, default_kwargs)]
        for method, kwargs in method_list:
            if method.supports_basis_gates:
                kwargs['basis_gates'] = self._basis_gates
            if method.supports_natural_direction:
                kwargs['natural_direction'] = self._natural_direction
            if method.supports_pulse_optimize:
                kwargs['pulse_optimize'] = self._pulse_optimize
            if method.supports_gate_lengths:
                _gate_lengths = _gate_lengths or _build_gate_lengths(self._backend_props, self._target)
                kwargs['gate_lengths'] = _gate_lengths
            if method.supports_gate_errors:
                _gate_errors = _gate_errors or _build_gate_errors(self._backend_props, self._target)
                kwargs['gate_errors'] = _gate_errors
            if method.supports_gate_lengths_by_qubit:
                _gate_lengths_by_qubit = _gate_lengths_by_qubit or _build_gate_lengths_by_qubit(self._backend_props, self._target)
                kwargs['gate_lengths_by_qubit'] = _gate_lengths_by_qubit
            if method.supports_gate_errors_by_qubit:
                _gate_errors_by_qubit = _gate_errors_by_qubit or _build_gate_errors_by_qubit(self._backend_props, self._target)
                kwargs['gate_errors_by_qubit'] = _gate_errors_by_qubit
            supported_bases = method.supported_bases
            if supported_bases is not None:
                kwargs['matched_basis'] = _choose_bases(self._basis_gates, supported_bases)
            if method.supports_target:
                kwargs['target'] = self._target
        default_method._approximation_degree = self._approximation_degree
        if self.method == 'default':
            plugin_method._approximation_degree = self._approximation_degree
        qubit_indices = {bit: i for i, bit in enumerate(dag.qubits)} if plugin_method.supports_coupling_map or default_method.supports_coupling_map else {}
        return self._run_main_loop(dag, qubit_indices, plugin_method, plugin_kwargs, default_method, default_kwargs)

    def _run_main_loop(self, dag, qubit_indices, plugin_method, plugin_kwargs, default_method, default_kwargs):
        """Inner loop for the optimizer, after all DAG-independent set-up has been completed."""
        for node in dag.op_nodes(ControlFlowOp):
            node.op = node.op.replace_blocks([dag_to_circuit(self._run_main_loop(circuit_to_dag(block), {inner: qubit_indices[outer] for inner, outer in zip(block.qubits, node.qargs)}, plugin_method, plugin_kwargs, default_method, default_kwargs), copy_operations=False) for block in node.op.blocks])
        for node in dag.named_nodes(*self._synth_gates):
            if self._min_qubits is not None and len(node.qargs) < self._min_qubits:
                continue
            synth_dag = None
            unitary = node.op.to_matrix()
            n_qubits = len(node.qargs)
            if plugin_method.max_qubits is not None and n_qubits > plugin_method.max_qubits or (plugin_method.min_qubits is not None and n_qubits < plugin_method.min_qubits):
                method, kwargs = (default_method, default_kwargs)
            else:
                method, kwargs = (plugin_method, plugin_kwargs)
            if method.supports_coupling_map:
                kwargs['coupling_map'] = (self._coupling_map, [qubit_indices[x] for x in node.qargs])
            synth_dag = method.run(unitary, **kwargs)
            if synth_dag is not None:
                dag.substitute_node_with_dag(node, synth_dag)
        return dag