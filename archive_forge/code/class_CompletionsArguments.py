from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class CompletionsArguments(BaseSchema):
    """
    Arguments for 'completions' request.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'frameId': {'type': 'integer', 'description': 'Returns completions in the scope of this stack frame. If not specified, the completions are returned for the global scope.'}, 'text': {'type': 'string', 'description': 'One or more source lines. Typically this is the text a user has typed into the debug console before he asked for completion.'}, 'column': {'type': 'integer', 'description': 'The character position for which to determine the completion proposals.'}, 'line': {'type': 'integer', 'description': 'An optional line for which to determine the completion proposals. If missing the first line of the text is assumed.'}}
    __refs__ = set()
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, text, column, frameId=None, line=None, update_ids_from_dap=False, **kwargs):
        """
        :param string text: One or more source lines. Typically this is the text a user has typed into the debug console before he asked for completion.
        :param integer column: The character position for which to determine the completion proposals.
        :param integer frameId: Returns completions in the scope of this stack frame. If not specified, the completions are returned for the global scope.
        :param integer line: An optional line for which to determine the completion proposals. If missing the first line of the text is assumed.
        """
        self.text = text
        self.column = column
        self.frameId = frameId
        self.line = line
        if update_ids_from_dap:
            self.frameId = self._translate_id_from_dap(self.frameId)
        self.kwargs = kwargs

    @classmethod
    def update_dict_ids_from_dap(cls, dct):
        if 'frameId' in dct:
            dct['frameId'] = cls._translate_id_from_dap(dct['frameId'])
        return dct

    def to_dict(self, update_ids_to_dap=False):
        text = self.text
        column = self.column
        frameId = self.frameId
        line = self.line
        if update_ids_to_dap:
            if frameId is not None:
                frameId = self._translate_id_to_dap(frameId)
        dct = {'text': text, 'column': column}
        if frameId is not None:
            dct['frameId'] = frameId
        if line is not None:
            dct['line'] = line
        dct.update(self.kwargs)
        return dct

    @classmethod
    def update_dict_ids_to_dap(cls, dct):
        if 'frameId' in dct:
            dct['frameId'] = cls._translate_id_to_dap(dct['frameId'])
        return dct