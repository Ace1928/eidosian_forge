from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class BreakpointLocationsArguments(BaseSchema):
    """
    Arguments for 'breakpointLocations' request.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'source': {'description': "The source location of the breakpoints; either 'source.path' or 'source.reference' must be specified.", 'type': 'Source'}, 'line': {'type': 'integer', 'description': 'Start line of range to search possible breakpoint locations in. If only the line is specified, the request returns all possible locations in that line.'}, 'column': {'type': 'integer', 'description': 'Optional start column of range to search possible breakpoint locations in. If no start column is given, the first column in the start line is assumed.'}, 'endLine': {'type': 'integer', 'description': 'Optional end line of range to search possible breakpoint locations in. If no end line is given, then the end line is assumed to be the start line.'}, 'endColumn': {'type': 'integer', 'description': 'Optional end column of range to search possible breakpoint locations in. If no end column is given, then it is assumed to be in the last column of the end line.'}}
    __refs__ = set(['source'])
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, source, line, column=None, endLine=None, endColumn=None, update_ids_from_dap=False, **kwargs):
        """
        :param Source source: The source location of the breakpoints; either 'source.path' or 'source.reference' must be specified.
        :param integer line: Start line of range to search possible breakpoint locations in. If only the line is specified, the request returns all possible locations in that line.
        :param integer column: Optional start column of range to search possible breakpoint locations in. If no start column is given, the first column in the start line is assumed.
        :param integer endLine: Optional end line of range to search possible breakpoint locations in. If no end line is given, then the end line is assumed to be the start line.
        :param integer endColumn: Optional end column of range to search possible breakpoint locations in. If no end column is given, then it is assumed to be in the last column of the end line.
        """
        if source is None:
            self.source = Source()
        else:
            self.source = Source(update_ids_from_dap=update_ids_from_dap, **source) if source.__class__ != Source else source
        self.line = line
        self.column = column
        self.endLine = endLine
        self.endColumn = endColumn
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        source = self.source
        line = self.line
        column = self.column
        endLine = self.endLine
        endColumn = self.endColumn
        dct = {'source': source.to_dict(update_ids_to_dap=update_ids_to_dap), 'line': line}
        if column is not None:
            dct['column'] = column
        if endLine is not None:
            dct['endLine'] = endLine
        if endColumn is not None:
            dct['endColumn'] = endColumn
        dct.update(self.kwargs)
        return dct