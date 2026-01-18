from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register_response('stepBack')
@register
class StepBackResponse(BaseSchema):
    """
    Response to 'stepBack' request. This is just an acknowledgement, so no body field is required.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'seq': {'type': 'integer', 'description': "Sequence number (also known as message ID). For protocol messages of type 'request' this ID can be used to cancel the request."}, 'type': {'type': 'string', 'enum': ['response']}, 'request_seq': {'type': 'integer', 'description': 'Sequence number of the corresponding request.'}, 'success': {'type': 'boolean', 'description': "Outcome of the request.\nIf true, the request was successful and the 'body' attribute may contain the result of the request.\nIf the value is false, the attribute 'message' contains the error in short form and the 'body' may contain additional information (see 'ErrorResponse.body.error')."}, 'command': {'type': 'string', 'description': 'The command requested.'}, 'message': {'type': 'string', 'description': "Contains the raw error in short form if 'success' is false.\nThis raw error might be interpreted by the frontend and is not shown in the UI.\nSome predefined values exist.", '_enum': ['cancelled'], 'enumDescriptions': ['request was cancelled.']}, 'body': {'type': ['array', 'boolean', 'integer', 'null', 'number', 'object', 'string'], 'description': 'Contains request result if success is true and optional error details if success is false.'}}
    __refs__ = set()
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, request_seq, success, command, seq=-1, message=None, body=None, update_ids_from_dap=False, **kwargs):
        """
        :param string type: 
        :param integer request_seq: Sequence number of the corresponding request.
        :param boolean success: Outcome of the request.
        If true, the request was successful and the 'body' attribute may contain the result of the request.
        If the value is false, the attribute 'message' contains the error in short form and the 'body' may contain additional information (see 'ErrorResponse.body.error').
        :param string command: The command requested.
        :param integer seq: Sequence number (also known as message ID). For protocol messages of type 'request' this ID can be used to cancel the request.
        :param string message: Contains the raw error in short form if 'success' is false.
        This raw error might be interpreted by the frontend and is not shown in the UI.
        Some predefined values exist.
        :param ['array', 'boolean', 'integer', 'null', 'number', 'object', 'string'] body: Contains request result if success is true and optional error details if success is false.
        """
        self.type = 'response'
        self.request_seq = request_seq
        self.success = success
        self.command = command
        self.seq = seq
        self.message = message
        self.body = body
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        type = self.type
        request_seq = self.request_seq
        success = self.success
        command = self.command
        seq = self.seq
        message = self.message
        body = self.body
        dct = {'type': type, 'request_seq': request_seq, 'success': success, 'command': command, 'seq': seq}
        if message is not None:
            dct['message'] = message
        if body is not None:
            dct['body'] = body
        dct.update(self.kwargs)
        return dct