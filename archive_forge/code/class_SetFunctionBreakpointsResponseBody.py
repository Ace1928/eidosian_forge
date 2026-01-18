from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class SetFunctionBreakpointsResponseBody(BaseSchema):
    """
    "body" of SetFunctionBreakpointsResponse

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'breakpoints': {'type': 'array', 'items': {'$ref': '#/definitions/Breakpoint'}, 'description': "Information about the breakpoints. The array elements correspond to the elements of the 'breakpoints' array."}}
    __refs__ = set()
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, breakpoints, update_ids_from_dap=False, **kwargs):
        """
        :param array breakpoints: Information about the breakpoints. The array elements correspond to the elements of the 'breakpoints' array.
        """
        self.breakpoints = breakpoints
        if update_ids_from_dap and self.breakpoints:
            for o in self.breakpoints:
                Breakpoint.update_dict_ids_from_dap(o)
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        breakpoints = self.breakpoints
        if breakpoints and hasattr(breakpoints[0], 'to_dict'):
            breakpoints = [x.to_dict() for x in breakpoints]
        dct = {'breakpoints': [Breakpoint.update_dict_ids_to_dap(o) for o in breakpoints] if update_ids_to_dap and breakpoints else breakpoints}
        dct.update(self.kwargs)
        return dct