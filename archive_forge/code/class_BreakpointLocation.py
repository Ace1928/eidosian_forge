from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class BreakpointLocation(BaseSchema):
    """
    Properties of a breakpoint location returned from the 'breakpointLocations' request.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'line': {'type': 'integer', 'description': 'Start line of breakpoint location.'}, 'column': {'type': 'integer', 'description': 'Optional start column of breakpoint location.'}, 'endLine': {'type': 'integer', 'description': 'Optional end line of breakpoint location if the location covers a range.'}, 'endColumn': {'type': 'integer', 'description': 'Optional end column of breakpoint location if the location covers a range.'}}
    __refs__ = set()
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, line, column=None, endLine=None, endColumn=None, update_ids_from_dap=False, **kwargs):
        """
        :param integer line: Start line of breakpoint location.
        :param integer column: Optional start column of breakpoint location.
        :param integer endLine: Optional end line of breakpoint location if the location covers a range.
        :param integer endColumn: Optional end column of breakpoint location if the location covers a range.
        """
        self.line = line
        self.column = column
        self.endLine = endLine
        self.endColumn = endColumn
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        line = self.line
        column = self.column
        endLine = self.endLine
        endColumn = self.endColumn
        dct = {'line': line}
        if column is not None:
            dct['column'] = column
        if endLine is not None:
            dct['endLine'] = endLine
        if endColumn is not None:
            dct['endColumn'] = endColumn
        dct.update(self.kwargs)
        return dct