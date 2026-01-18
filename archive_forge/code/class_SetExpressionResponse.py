from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register_response('setExpression')
@register
class SetExpressionResponse(BaseSchema):
    """
    Response to 'setExpression' request.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'seq': {'type': 'integer', 'description': "Sequence number (also known as message ID). For protocol messages of type 'request' this ID can be used to cancel the request."}, 'type': {'type': 'string', 'enum': ['response']}, 'request_seq': {'type': 'integer', 'description': 'Sequence number of the corresponding request.'}, 'success': {'type': 'boolean', 'description': "Outcome of the request.\nIf true, the request was successful and the 'body' attribute may contain the result of the request.\nIf the value is false, the attribute 'message' contains the error in short form and the 'body' may contain additional information (see 'ErrorResponse.body.error')."}, 'command': {'type': 'string', 'description': 'The command requested.'}, 'message': {'type': 'string', 'description': "Contains the raw error in short form if 'success' is false.\nThis raw error might be interpreted by the frontend and is not shown in the UI.\nSome predefined values exist.", '_enum': ['cancelled'], 'enumDescriptions': ['request was cancelled.']}, 'body': {'type': 'object', 'properties': {'value': {'type': 'string', 'description': 'The new value of the expression.'}, 'type': {'type': 'string', 'description': "The optional type of the value.\nThis attribute should only be returned by a debug adapter if the client has passed the value true for the 'supportsVariableType' capability of the 'initialize' request."}, 'presentationHint': {'$ref': '#/definitions/VariablePresentationHint', 'description': 'Properties of a value that can be used to determine how to render the result in the UI.'}, 'variablesReference': {'type': 'integer', 'description': 'If variablesReference is > 0, the value is structured and its children can be retrieved by passing variablesReference to the VariablesRequest.\nThe value should be less than or equal to 2147483647 (2^31-1).'}, 'namedVariables': {'type': 'integer', 'description': 'The number of named child variables.\nThe client can use this optional information to present the variables in a paged UI and fetch them in chunks.\nThe value should be less than or equal to 2147483647 (2^31-1).'}, 'indexedVariables': {'type': 'integer', 'description': 'The number of indexed child variables.\nThe client can use this optional information to present the variables in a paged UI and fetch them in chunks.\nThe value should be less than or equal to 2147483647 (2^31-1).'}}, 'required': ['value']}}
    __refs__ = set(['body'])
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, request_seq, success, command, body, seq=-1, message=None, update_ids_from_dap=False, **kwargs):
        """
        :param string type: 
        :param integer request_seq: Sequence number of the corresponding request.
        :param boolean success: Outcome of the request.
        If true, the request was successful and the 'body' attribute may contain the result of the request.
        If the value is false, the attribute 'message' contains the error in short form and the 'body' may contain additional information (see 'ErrorResponse.body.error').
        :param string command: The command requested.
        :param SetExpressionResponseBody body: 
        :param integer seq: Sequence number (also known as message ID). For protocol messages of type 'request' this ID can be used to cancel the request.
        :param string message: Contains the raw error in short form if 'success' is false.
        This raw error might be interpreted by the frontend and is not shown in the UI.
        Some predefined values exist.
        """
        self.type = 'response'
        self.request_seq = request_seq
        self.success = success
        self.command = command
        if body is None:
            self.body = SetExpressionResponseBody()
        else:
            self.body = SetExpressionResponseBody(update_ids_from_dap=update_ids_from_dap, **body) if body.__class__ != SetExpressionResponseBody else body
        self.seq = seq
        self.message = message
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        type = self.type
        request_seq = self.request_seq
        success = self.success
        command = self.command
        body = self.body
        seq = self.seq
        message = self.message
        dct = {'type': type, 'request_seq': request_seq, 'success': success, 'command': command, 'body': body.to_dict(update_ids_to_dap=update_ids_to_dap), 'seq': seq}
        if message is not None:
            dct['message'] = message
        dct.update(self.kwargs)
        return dct