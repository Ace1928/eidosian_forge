from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class GotoTargetsArguments(BaseSchema):
    """
    Arguments for 'gotoTargets' request.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'source': {'description': 'The source location for which the goto targets are determined.', 'type': 'Source'}, 'line': {'type': 'integer', 'description': 'The line location for which the goto targets are determined.'}, 'column': {'type': 'integer', 'description': 'An optional column location for which the goto targets are determined.'}}
    __refs__ = set(['source'])
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, source, line, column=None, update_ids_from_dap=False, **kwargs):
        """
        :param Source source: The source location for which the goto targets are determined.
        :param integer line: The line location for which the goto targets are determined.
        :param integer column: An optional column location for which the goto targets are determined.
        """
        if source is None:
            self.source = Source()
        else:
            self.source = Source(update_ids_from_dap=update_ids_from_dap, **source) if source.__class__ != Source else source
        self.line = line
        self.column = column
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        source = self.source
        line = self.line
        column = self.column
        dct = {'source': source.to_dict(update_ids_to_dap=update_ids_to_dap), 'line': line}
        if column is not None:
            dct['column'] = column
        dct.update(self.kwargs)
        return dct