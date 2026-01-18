from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class SourceBreakpoint(BaseSchema):
    """
    Properties of a breakpoint or logpoint passed to the setBreakpoints request.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'line': {'type': 'integer', 'description': 'The source line of the breakpoint or logpoint.'}, 'column': {'type': 'integer', 'description': 'An optional source column of the breakpoint.'}, 'condition': {'type': 'string', 'description': "An optional expression for conditional breakpoints.\nIt is only honored by a debug adapter if the capability 'supportsConditionalBreakpoints' is true."}, 'hitCondition': {'type': 'string', 'description': "An optional expression that controls how many hits of the breakpoint are ignored.\nThe backend is expected to interpret the expression as needed.\nThe attribute is only honored by a debug adapter if the capability 'supportsHitConditionalBreakpoints' is true."}, 'logMessage': {'type': 'string', 'description': "If this attribute exists and is non-empty, the backend must not 'break' (stop)\nbut log the message instead. Expressions within {} are interpolated.\nThe attribute is only honored by a debug adapter if the capability 'supportsLogPoints' is true."}}
    __refs__ = set()
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, line, column=None, condition=None, hitCondition=None, logMessage=None, update_ids_from_dap=False, **kwargs):
        """
        :param integer line: The source line of the breakpoint or logpoint.
        :param integer column: An optional source column of the breakpoint.
        :param string condition: An optional expression for conditional breakpoints.
        It is only honored by a debug adapter if the capability 'supportsConditionalBreakpoints' is true.
        :param string hitCondition: An optional expression that controls how many hits of the breakpoint are ignored.
        The backend is expected to interpret the expression as needed.
        The attribute is only honored by a debug adapter if the capability 'supportsHitConditionalBreakpoints' is true.
        :param string logMessage: If this attribute exists and is non-empty, the backend must not 'break' (stop)
        but log the message instead. Expressions within {} are interpolated.
        The attribute is only honored by a debug adapter if the capability 'supportsLogPoints' is true.
        """
        self.line = line
        self.column = column
        self.condition = condition
        self.hitCondition = hitCondition
        self.logMessage = logMessage
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        line = self.line
        column = self.column
        condition = self.condition
        hitCondition = self.hitCondition
        logMessage = self.logMessage
        dct = {'line': line}
        if column is not None:
            dct['column'] = column
        if condition is not None:
            dct['condition'] = condition
        if hitCondition is not None:
            dct['hitCondition'] = hitCondition
        if logMessage is not None:
            dct['logMessage'] = logMessage
        dct.update(self.kwargs)
        return dct