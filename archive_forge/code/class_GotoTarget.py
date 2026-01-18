from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class GotoTarget(BaseSchema):
    """
    A GotoTarget describes a code location that can be used as a target in the 'goto' request.
    
    The possible goto targets can be determined via the 'gotoTargets' request.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'id': {'type': 'integer', 'description': 'Unique identifier for a goto target. This is used in the goto request.'}, 'label': {'type': 'string', 'description': 'The name of the goto target (shown in the UI).'}, 'line': {'type': 'integer', 'description': 'The line of the goto target.'}, 'column': {'type': 'integer', 'description': 'An optional column of the goto target.'}, 'endLine': {'type': 'integer', 'description': 'An optional end line of the range covered by the goto target.'}, 'endColumn': {'type': 'integer', 'description': 'An optional end column of the range covered by the goto target.'}, 'instructionPointerReference': {'type': 'string', 'description': 'Optional memory reference for the instruction pointer value represented by this target.'}}
    __refs__ = set()
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, id, label, line, column=None, endLine=None, endColumn=None, instructionPointerReference=None, update_ids_from_dap=False, **kwargs):
        """
        :param integer id: Unique identifier for a goto target. This is used in the goto request.
        :param string label: The name of the goto target (shown in the UI).
        :param integer line: The line of the goto target.
        :param integer column: An optional column of the goto target.
        :param integer endLine: An optional end line of the range covered by the goto target.
        :param integer endColumn: An optional end column of the range covered by the goto target.
        :param string instructionPointerReference: Optional memory reference for the instruction pointer value represented by this target.
        """
        self.id = id
        self.label = label
        self.line = line
        self.column = column
        self.endLine = endLine
        self.endColumn = endColumn
        self.instructionPointerReference = instructionPointerReference
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        id = self.id
        label = self.label
        line = self.line
        column = self.column
        endLine = self.endLine
        endColumn = self.endColumn
        instructionPointerReference = self.instructionPointerReference
        dct = {'id': id, 'label': label, 'line': line}
        if column is not None:
            dct['column'] = column
        if endLine is not None:
            dct['endLine'] = endLine
        if endColumn is not None:
            dct['endColumn'] = endColumn
        if instructionPointerReference is not None:
            dct['instructionPointerReference'] = instructionPointerReference
        dct.update(self.kwargs)
        return dct