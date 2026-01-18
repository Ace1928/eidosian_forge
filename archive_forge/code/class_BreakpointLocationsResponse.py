from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register_response('breakpointLocations')
@register
class BreakpointLocationsResponse(BaseSchema):
    """
    Response to 'breakpointLocations' request.
    
    Contains possible locations for source breakpoints.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'seq': {'type': 'integer', 'description': "Sequence number (also known as message ID). For protocol messages of type 'request' this ID can be used to cancel the request."}, 'type': {'type': 'string', 'enum': ['response']}, 'request_seq': {'type': 'integer', 'description': 'Sequence number of the corresponding request.'}, 'success': {'type': 'boolean', 'description': "Outcome of the request.\nIf true, the request was successful and the 'body' attribute may contain the result of the request.\nIf the value is false, the attribute 'message' contains the error in short form and the 'body' may contain additional information (see 'ErrorResponse.body.error')."}, 'command': {'type': 'string', 'description': 'The command requested.'}, 'message': {'type': 'string', 'description': "Contains the raw error in short form if 'success' is false.\nThis raw error might be interpreted by the frontend and is not shown in the UI.\nSome predefined values exist.", '_enum': ['cancelled'], 'enumDescriptions': ['request was cancelled.']}, 'body': {'type': 'object', 'properties': {'breakpoints': {'type': 'array', 'items': {'$ref': '#/definitions/BreakpointLocation'}, 'description': 'Sorted set of possible breakpoint locations.'}}, 'required': ['breakpoints']}}
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
        :param BreakpointLocationsResponseBody body: 
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
            self.body = BreakpointLocationsResponseBody()
        else:
            self.body = BreakpointLocationsResponseBody(update_ids_from_dap=update_ids_from_dap, **body) if body.__class__ != BreakpointLocationsResponseBody else body
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