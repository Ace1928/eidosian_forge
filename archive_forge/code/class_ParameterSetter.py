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
class ParameterSetter(NodeVisitor):
    """Node visitor for parameter binding.

    This visitor is initialized with a dictionary of parameters to be assigned,
    and assign values to operands of nodes found.
    """

    def __init__(self, param_map: dict[ParameterExpression, ParameterValueType]):
        self._param_map = param_map

    def visit_ScheduleBlock(self, node: ScheduleBlock):
        """Visit ``ScheduleBlock``. Recursively visit context blocks and overwrite.

        .. note:: ``ScheduleBlock`` can have parameters in blocks and its alignment.
        """
        node._alignment_context = self.visit_AlignmentKind(node.alignment_context)
        for elm in node._blocks:
            self.visit(elm)
        self._update_parameter_manager(node)
        return node

    def visit_Schedule(self, node: Schedule):
        """Visit ``Schedule``. Recursively visit schedule children and overwrite."""
        node._Schedule__children = [(t0, self.visit(sched)) for t0, sched in node.instructions]
        node._renew_timeslots()
        self._update_parameter_manager(node)
        return node

    def visit_AlignmentKind(self, node: AlignmentKind):
        """Assign parameters to block's ``AlignmentKind`` specification."""
        new_parameters = tuple((self.visit(param) for param in node._context_params))
        node._context_params = new_parameters
        return node

    def visit_Instruction(self, node: instructions.Instruction):
        """Assign parameters to general pulse instruction.

        .. note:: All parametrized object should be stored in the operands.
            Otherwise parameter cannot be detected.
        """
        if node.is_parameterized():
            node._operands = tuple((self.visit(op) for op in node.operands))
        return node

    def visit_Channel(self, node: channels.Channel):
        """Assign parameters to ``Channel`` object."""
        if node.is_parameterized():
            new_index = self._assign_parameter_expression(node.index)
            if not isinstance(new_index, ParameterExpression):
                if not isinstance(new_index, int) or new_index < 0:
                    raise PulseError('Channel index must be a nonnegative integer')
            return node.__class__(index=new_index)
        return node

    def visit_SymbolicPulse(self, node: SymbolicPulse):
        """Assign parameters to ``SymbolicPulse`` object."""
        if node.is_parameterized():
            if isinstance(node.duration, ParameterExpression):
                node.duration = self._assign_parameter_expression(node.duration)
            for name in node._params:
                pval = node._params[name]
                if isinstance(pval, ParameterExpression):
                    new_val = self._assign_parameter_expression(pval)
                    node._params[name] = new_val
            if not node.disable_validation:
                node.validate_parameters()
        return node

    def visit_Waveform(self, node: Waveform):
        """Assign parameters to ``Waveform`` object.

        .. node:: No parameter can be assigned to ``Waveform`` object.
        """
        return node

    def generic_visit(self, node: Any):
        """Assign parameters to object that doesn't belong to Qiskit Pulse module."""
        if isinstance(node, ParameterExpression):
            return self._assign_parameter_expression(node)
        else:
            return node

    def _assign_parameter_expression(self, param_expr: ParameterExpression):
        """A helper function to assign parameter value to parameter expression."""
        new_value = copy(param_expr)
        updated = param_expr.parameters & self._param_map.keys()
        for param in updated:
            new_value = new_value.assign(param, self._param_map[param])
        new_value = format_parameter_value(new_value)
        return new_value

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