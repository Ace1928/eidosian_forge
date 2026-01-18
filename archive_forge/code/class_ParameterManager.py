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
class ParameterManager:
    """Helper class to manage parameter objects associated with arbitrary pulse programs.

    This object is implicitly initialized with the parameter object storage
    that stores parameter objects added to the parent pulse program.

    Parameter assignment logic is implemented based on the visitor pattern.
    Instruction data and its location are not directly associated with this object.
    """

    def __init__(self):
        """Create new parameter table for pulse programs."""
        self._parameters = set()

    @property
    def parameters(self) -> set[Parameter]:
        """Parameters which determine the schedule behavior."""
        return self._parameters

    def clear(self):
        """Remove the parameters linked to this manager."""
        self._parameters.clear()

    def is_parameterized(self) -> bool:
        """Return True iff the instruction is parameterized."""
        return bool(self.parameters)

    def get_parameters(self, parameter_name: str) -> list[Parameter]:
        """Get parameter object bound to this schedule by string name.

        Because different ``Parameter`` objects can have the same name,
        this method returns a list of ``Parameter`` s for the provided name.

        Args:
            parameter_name: Name of parameter.

        Returns:
            Parameter objects that have corresponding name.
        """
        return [param for param in self.parameters if param.name == parameter_name]

    def assign_parameters(self, pulse_program: Any, value_dict: dict[ParameterExpression, ParameterValueType]) -> Any:
        """Modify and return program data with parameters assigned according to the input.

        Args:
            pulse_program: Arbitrary pulse program associated with this manager instance.
            value_dict: A mapping from Parameters to either numeric values or another
                Parameter expression.

        Returns:
            Updated program data.
        """
        valid_map = {k: value_dict[k] for k in value_dict.keys() & self._parameters}
        if valid_map:
            visitor = ParameterSetter(param_map=valid_map)
            return visitor.visit(pulse_program)
        return pulse_program

    def update_parameter_table(self, new_node: Any):
        """A helper function to update parameter table with given data node.

        Args:
            new_node: A new data node to be added.
        """
        visitor = ParameterGetter()
        visitor.visit(new_node)
        self._parameters |= visitor.parameters