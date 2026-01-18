from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class EvaluateResponseBody(BaseSchema):
    """
    "body" of EvaluateResponse

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'result': {'type': 'string', 'description': 'The result of the evaluate request.'}, 'type': {'type': 'string', 'description': "The optional type of the evaluate result.\nThis attribute should only be returned by a debug adapter if the client has passed the value true for the 'supportsVariableType' capability of the 'initialize' request."}, 'presentationHint': {'description': 'Properties of a evaluate result that can be used to determine how to render the result in the UI.', 'type': 'VariablePresentationHint'}, 'variablesReference': {'type': 'integer', 'description': 'If variablesReference is > 0, the evaluate result is structured and its children can be retrieved by passing variablesReference to the VariablesRequest.\nThe value should be less than or equal to 2147483647 (2^31-1).'}, 'namedVariables': {'type': 'integer', 'description': 'The number of named child variables.\nThe client can use this optional information to present the variables in a paged UI and fetch them in chunks.\nThe value should be less than or equal to 2147483647 (2^31-1).'}, 'indexedVariables': {'type': 'integer', 'description': 'The number of indexed child variables.\nThe client can use this optional information to present the variables in a paged UI and fetch them in chunks.\nThe value should be less than or equal to 2147483647 (2^31-1).'}, 'memoryReference': {'type': 'string', 'description': "Optional memory reference to a location appropriate for this result.\nFor pointer type eval results, this is generally a reference to the memory address contained in the pointer.\nThis attribute should be returned by a debug adapter if the client has passed the value true for the 'supportsMemoryReferences' capability of the 'initialize' request."}}
    __refs__ = set(['presentationHint'])
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, result, variablesReference, type=None, presentationHint=None, namedVariables=None, indexedVariables=None, memoryReference=None, update_ids_from_dap=False, **kwargs):
        """
        :param string result: The result of the evaluate request.
        :param integer variablesReference: If variablesReference is > 0, the evaluate result is structured and its children can be retrieved by passing variablesReference to the VariablesRequest.
        The value should be less than or equal to 2147483647 (2^31-1).
        :param string type: The optional type of the evaluate result.
        This attribute should only be returned by a debug adapter if the client has passed the value true for the 'supportsVariableType' capability of the 'initialize' request.
        :param VariablePresentationHint presentationHint: Properties of a evaluate result that can be used to determine how to render the result in the UI.
        :param integer namedVariables: The number of named child variables.
        The client can use this optional information to present the variables in a paged UI and fetch them in chunks.
        The value should be less than or equal to 2147483647 (2^31-1).
        :param integer indexedVariables: The number of indexed child variables.
        The client can use this optional information to present the variables in a paged UI and fetch them in chunks.
        The value should be less than or equal to 2147483647 (2^31-1).
        :param string memoryReference: Optional memory reference to a location appropriate for this result.
        For pointer type eval results, this is generally a reference to the memory address contained in the pointer.
        This attribute should be returned by a debug adapter if the client has passed the value true for the 'supportsMemoryReferences' capability of the 'initialize' request.
        """
        self.result = result
        self.variablesReference = variablesReference
        self.type = type
        if presentationHint is None:
            self.presentationHint = VariablePresentationHint()
        else:
            self.presentationHint = VariablePresentationHint(update_ids_from_dap=update_ids_from_dap, **presentationHint) if presentationHint.__class__ != VariablePresentationHint else presentationHint
        self.namedVariables = namedVariables
        self.indexedVariables = indexedVariables
        self.memoryReference = memoryReference
        if update_ids_from_dap:
            self.variablesReference = self._translate_id_from_dap(self.variablesReference)
        self.kwargs = kwargs

    @classmethod
    def update_dict_ids_from_dap(cls, dct):
        if 'variablesReference' in dct:
            dct['variablesReference'] = cls._translate_id_from_dap(dct['variablesReference'])
        return dct

    def to_dict(self, update_ids_to_dap=False):
        result = self.result
        variablesReference = self.variablesReference
        type = self.type
        presentationHint = self.presentationHint
        namedVariables = self.namedVariables
        indexedVariables = self.indexedVariables
        memoryReference = self.memoryReference
        if update_ids_to_dap:
            if variablesReference is not None:
                variablesReference = self._translate_id_to_dap(variablesReference)
        dct = {'result': result, 'variablesReference': variablesReference}
        if type is not None:
            dct['type'] = type
        if presentationHint is not None:
            dct['presentationHint'] = presentationHint.to_dict(update_ids_to_dap=update_ids_to_dap)
        if namedVariables is not None:
            dct['namedVariables'] = namedVariables
        if indexedVariables is not None:
            dct['indexedVariables'] = indexedVariables
        if memoryReference is not None:
            dct['memoryReference'] = memoryReference
        dct.update(self.kwargs)
        return dct

    @classmethod
    def update_dict_ids_to_dap(cls, dct):
        if 'variablesReference' in dct:
            dct['variablesReference'] = cls._translate_id_to_dap(dct['variablesReference'])
        return dct