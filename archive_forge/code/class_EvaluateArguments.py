from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class EvaluateArguments(BaseSchema):
    """
    Arguments for 'evaluate' request.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'expression': {'type': 'string', 'description': 'The expression to evaluate.'}, 'frameId': {'type': 'integer', 'description': 'Evaluate the expression in the scope of this stack frame. If not specified, the expression is evaluated in the global scope.'}, 'context': {'type': 'string', '_enum': ['watch', 'repl', 'hover', 'clipboard'], 'enumDescriptions': ['evaluate is run in a watch.', 'evaluate is run from REPL console.', 'evaluate is run from a data hover.', "evaluate is run to generate the value that will be stored in the clipboard.\nThe attribute is only honored by a debug adapter if the capability 'supportsClipboardContext' is true."], 'description': 'The context in which the evaluate request is run.'}, 'format': {'description': "Specifies details on how to format the Evaluate result.\nThe attribute is only honored by a debug adapter if the capability 'supportsValueFormattingOptions' is true.", 'type': 'ValueFormat'}}
    __refs__ = set(['format'])
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, expression, frameId=None, context=None, format=None, update_ids_from_dap=False, **kwargs):
        """
        :param string expression: The expression to evaluate.
        :param integer frameId: Evaluate the expression in the scope of this stack frame. If not specified, the expression is evaluated in the global scope.
        :param string context: The context in which the evaluate request is run.
        :param ValueFormat format: Specifies details on how to format the Evaluate result.
        The attribute is only honored by a debug adapter if the capability 'supportsValueFormattingOptions' is true.
        """
        self.expression = expression
        self.frameId = frameId
        self.context = context
        if format is None:
            self.format = ValueFormat()
        else:
            self.format = ValueFormat(update_ids_from_dap=update_ids_from_dap, **format) if format.__class__ != ValueFormat else format
        if update_ids_from_dap:
            self.frameId = self._translate_id_from_dap(self.frameId)
        self.kwargs = kwargs

    @classmethod
    def update_dict_ids_from_dap(cls, dct):
        if 'frameId' in dct:
            dct['frameId'] = cls._translate_id_from_dap(dct['frameId'])
        return dct

    def to_dict(self, update_ids_to_dap=False):
        expression = self.expression
        frameId = self.frameId
        context = self.context
        format = self.format
        if update_ids_to_dap:
            if frameId is not None:
                frameId = self._translate_id_to_dap(frameId)
        dct = {'expression': expression}
        if frameId is not None:
            dct['frameId'] = frameId
        if context is not None:
            dct['context'] = context
        if format is not None:
            dct['format'] = format.to_dict(update_ids_to_dap=update_ids_to_dap)
        dct.update(self.kwargs)
        return dct

    @classmethod
    def update_dict_ids_to_dap(cls, dct):
        if 'frameId' in dct:
            dct['frameId'] = cls._translate_id_to_dap(dct['frameId'])
        return dct