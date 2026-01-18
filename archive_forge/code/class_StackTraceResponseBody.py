from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class StackTraceResponseBody(BaseSchema):
    """
    "body" of StackTraceResponse

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'stackFrames': {'type': 'array', 'items': {'$ref': '#/definitions/StackFrame'}, 'description': 'The frames of the stackframe. If the array has length zero, there are no stackframes available.\nThis means that there is no location information available.'}, 'totalFrames': {'type': 'integer', 'description': 'The total number of frames available in the stack. If omitted or if totalFrames is larger than the available frames, a client is expected to request frames until a request returns less frames than requested (which indicates the end of the stack). Returning monotonically increasing totalFrames values for subsequent requests can be used to enforce paging in the client.'}}
    __refs__ = set()
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, stackFrames, totalFrames=None, update_ids_from_dap=False, **kwargs):
        """
        :param array stackFrames: The frames of the stackframe. If the array has length zero, there are no stackframes available.
        This means that there is no location information available.
        :param integer totalFrames: The total number of frames available in the stack. If omitted or if totalFrames is larger than the available frames, a client is expected to request frames until a request returns less frames than requested (which indicates the end of the stack). Returning monotonically increasing totalFrames values for subsequent requests can be used to enforce paging in the client.
        """
        self.stackFrames = stackFrames
        if update_ids_from_dap and self.stackFrames:
            for o in self.stackFrames:
                StackFrame.update_dict_ids_from_dap(o)
        self.totalFrames = totalFrames
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        stackFrames = self.stackFrames
        if stackFrames and hasattr(stackFrames[0], 'to_dict'):
            stackFrames = [x.to_dict() for x in stackFrames]
        totalFrames = self.totalFrames
        dct = {'stackFrames': [StackFrame.update_dict_ids_to_dap(o) for o in stackFrames] if update_ids_to_dap and stackFrames else stackFrames}
        if totalFrames is not None:
            dct['totalFrames'] = totalFrames
        dct.update(self.kwargs)
        return dct