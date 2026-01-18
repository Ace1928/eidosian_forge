from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class ExceptionFilterOptions(BaseSchema):
    """
    An ExceptionFilterOptions is used to specify an exception filter together with a condition for the
    setExceptionsFilter request.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'filterId': {'type': 'string', 'description': "ID of an exception filter returned by the 'exceptionBreakpointFilters' capability."}, 'condition': {'type': 'string', 'description': 'An optional expression for conditional exceptions.\nThe exception will break into the debugger if the result of the condition is true.'}}
    __refs__ = set()
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, filterId, condition=None, update_ids_from_dap=False, **kwargs):
        """
        :param string filterId: ID of an exception filter returned by the 'exceptionBreakpointFilters' capability.
        :param string condition: An optional expression for conditional exceptions.
        The exception will break into the debugger if the result of the condition is true.
        """
        self.filterId = filterId
        self.condition = condition
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        filterId = self.filterId
        condition = self.condition
        dct = {'filterId': filterId}
        if condition is not None:
            dct['condition'] = condition
        dct.update(self.kwargs)
        return dct