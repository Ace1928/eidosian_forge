from a backend
from __future__ import annotations
import itertools
from typing import Optional, List, Any
from collections.abc import Mapping
from collections import defaultdict
import datetime
import io
import logging
import inspect
import rustworkx as rx
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.circuit.gate import Gate
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
from qiskit.pulse.instruction_schedule_map import InstructionScheduleMap
from qiskit.pulse.calibration_entries import CalibrationEntry, ScheduleDef
from qiskit.pulse.schedule import Schedule, ScheduleBlock
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.instruction_durations import InstructionDurations
from qiskit.transpiler.timing_constraints import TimingConstraints
from qiskit.providers.exceptions import BackendPropertyError
from qiskit.pulse.exceptions import PulseError, UnassignedDurationError
from qiskit.exceptions import QiskitError
from qiskit.providers.backend import QubitProperties  # pylint: disable=unused-import
from qiskit.providers.models.backendproperties import BackendProperties
def _build_coupling_graph(self):
    self._coupling_graph = rx.PyDiGraph(multigraph=False)
    self._coupling_graph.add_nodes_from([{} for _ in range(self.num_qubits)])
    for gate, qarg_map in self._gate_map.items():
        if qarg_map is None:
            if self._gate_name_map[gate].num_qubits == 2:
                self._coupling_graph = None
                return
            continue
        for qarg, properties in qarg_map.items():
            if qarg is None:
                if self._gate_name_map[gate].num_qubits == 2:
                    self._coupling_graph = None
                    return
                continue
            if len(qarg) == 1:
                self._coupling_graph[qarg[0]] = properties
            elif len(qarg) == 2:
                try:
                    edge_data = self._coupling_graph.get_edge_data(*qarg)
                    edge_data[gate] = properties
                except rx.NoEdgeBetweenNodes:
                    self._coupling_graph.add_edge(*qarg, {gate: properties})
    if self._coupling_graph.num_edges() == 0 and any((x is None for x in self._qarg_gate_map)):
        self._coupling_graph = None