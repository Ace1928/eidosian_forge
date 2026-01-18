from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class SetBreakpointsArguments(BaseSchema):
    """
    Arguments for 'setBreakpoints' request.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'source': {'description': "The source location of the breakpoints; either 'source.path' or 'source.reference' must be specified.", 'type': 'Source'}, 'breakpoints': {'type': 'array', 'items': {'$ref': '#/definitions/SourceBreakpoint'}, 'description': 'The code locations of the breakpoints.'}, 'lines': {'type': 'array', 'items': {'type': 'integer'}, 'description': 'Deprecated: The code locations of the breakpoints.'}, 'sourceModified': {'type': 'boolean', 'description': 'A value of true indicates that the underlying source has been modified which results in new breakpoint locations.'}}
    __refs__ = set(['source'])
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, source, breakpoints=None, lines=None, sourceModified=None, update_ids_from_dap=False, **kwargs):
        """
        :param Source source: The source location of the breakpoints; either 'source.path' or 'source.reference' must be specified.
        :param array breakpoints: The code locations of the breakpoints.
        :param array lines: Deprecated: The code locations of the breakpoints.
        :param boolean sourceModified: A value of true indicates that the underlying source has been modified which results in new breakpoint locations.
        """
        if source is None:
            self.source = Source()
        else:
            self.source = Source(update_ids_from_dap=update_ids_from_dap, **source) if source.__class__ != Source else source
        self.breakpoints = breakpoints
        if update_ids_from_dap and self.breakpoints:
            for o in self.breakpoints:
                SourceBreakpoint.update_dict_ids_from_dap(o)
        self.lines = lines
        self.sourceModified = sourceModified
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        source = self.source
        breakpoints = self.breakpoints
        if breakpoints and hasattr(breakpoints[0], 'to_dict'):
            breakpoints = [x.to_dict() for x in breakpoints]
        lines = self.lines
        if lines and hasattr(lines[0], 'to_dict'):
            lines = [x.to_dict() for x in lines]
        sourceModified = self.sourceModified
        dct = {'source': source.to_dict(update_ids_to_dap=update_ids_to_dap)}
        if breakpoints is not None:
            dct['breakpoints'] = [SourceBreakpoint.update_dict_ids_to_dap(o) for o in breakpoints] if update_ids_to_dap and breakpoints else breakpoints
        if lines is not None:
            dct['lines'] = lines
        if sourceModified is not None:
            dct['sourceModified'] = sourceModified
        dct.update(self.kwargs)
        return dct