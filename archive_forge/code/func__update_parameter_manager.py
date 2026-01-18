from the parameter table model of ~O(1), however, usually, this calculation occurs
from each object, yielding smaller object creation cost and higher performance
from __future__ import annotations
from copy import copy
from typing import Any
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.parameterexpression import ParameterExpression, ParameterValueType
from qiskit.pulse import instructions, channels
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.library import SymbolicPulse, Waveform
from qiskit.pulse.schedule import Schedule, ScheduleBlock
from qiskit.pulse.transforms.alignments import AlignmentKind
from qiskit.pulse.utils import format_parameter_value
def _update_parameter_manager(self, node: Schedule | ScheduleBlock):
    """A helper function to update parameter manager of pulse program."""
    if not hasattr(node, '_parameter_manager'):
        raise PulseError(f'Node type {node.__class__.__name__} has no parameter manager.')
    param_manager = node._parameter_manager
    updated = param_manager.parameters & self._param_map.keys()
    new_parameters = set()
    for param in param_manager.parameters:
        if param not in updated:
            new_parameters.add(param)
            continue
        new_value = self._param_map[param]
        if isinstance(new_value, ParameterExpression):
            new_parameters |= new_value.parameters
    param_manager._parameters = new_parameters