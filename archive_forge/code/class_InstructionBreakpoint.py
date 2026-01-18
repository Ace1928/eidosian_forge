from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class InstructionBreakpoint(BaseSchema):
    """
    Properties of a breakpoint passed to the setInstructionBreakpoints request

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'instructionReference': {'type': 'string', 'description': 'The instruction reference of the breakpoint.\nThis should be a memory or instruction pointer reference from an EvaluateResponse, Variable, StackFrame, GotoTarget, or Breakpoint.'}, 'offset': {'type': 'integer', 'description': 'An optional offset from the instruction reference.\nThis can be negative.'}, 'condition': {'type': 'string', 'description': "An optional expression for conditional breakpoints.\nIt is only honored by a debug adapter if the capability 'supportsConditionalBreakpoints' is true."}, 'hitCondition': {'type': 'string', 'description': "An optional expression that controls how many hits of the breakpoint are ignored.\nThe backend is expected to interpret the expression as needed.\nThe attribute is only honored by a debug adapter if the capability 'supportsHitConditionalBreakpoints' is true."}}
    __refs__ = set()
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, instructionReference, offset=None, condition=None, hitCondition=None, update_ids_from_dap=False, **kwargs):
        """
        :param string instructionReference: The instruction reference of the breakpoint.
        This should be a memory or instruction pointer reference from an EvaluateResponse, Variable, StackFrame, GotoTarget, or Breakpoint.
        :param integer offset: An optional offset from the instruction reference.
        This can be negative.
        :param string condition: An optional expression for conditional breakpoints.
        It is only honored by a debug adapter if the capability 'supportsConditionalBreakpoints' is true.
        :param string hitCondition: An optional expression that controls how many hits of the breakpoint are ignored.
        The backend is expected to interpret the expression as needed.
        The attribute is only honored by a debug adapter if the capability 'supportsHitConditionalBreakpoints' is true.
        """
        self.instructionReference = instructionReference
        self.offset = offset
        self.condition = condition
        self.hitCondition = hitCondition
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        instructionReference = self.instructionReference
        offset = self.offset
        condition = self.condition
        hitCondition = self.hitCondition
        dct = {'instructionReference': instructionReference}
        if offset is not None:
            dct['offset'] = offset
        if condition is not None:
            dct['condition'] = condition
        if hitCondition is not None:
            dct['hitCondition'] = hitCondition
        dct.update(self.kwargs)
        return dct