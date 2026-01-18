from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register_event('progressUpdate')
@register
class ProgressUpdateEvent(BaseSchema):
    """
    The event signals that the progress reporting needs to updated with a new message and/or percentage.
    
    The client does not have to update the UI immediately, but the clients needs to keep track of the
    message and/or percentage values.
    
    This event should only be sent if the client has passed the value true for the
    'supportsProgressReporting' capability of the 'initialize' request.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'seq': {'type': 'integer', 'description': "Sequence number (also known as message ID). For protocol messages of type 'request' this ID can be used to cancel the request."}, 'type': {'type': 'string', 'enum': ['event']}, 'event': {'type': 'string', 'enum': ['progressUpdate']}, 'body': {'type': 'object', 'properties': {'progressId': {'type': 'string', 'description': "The ID that was introduced in the initial 'progressStart' event."}, 'message': {'type': 'string', 'description': 'Optional, more detailed progress message. If omitted, the previous message (if any) is used.'}, 'percentage': {'type': 'number', 'description': 'Optional progress percentage to display (value range: 0 to 100). If omitted no percentage will be shown.'}}, 'required': ['progressId']}}
    __refs__ = set(['body'])
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, body, seq=-1, update_ids_from_dap=False, **kwargs):
        """
        :param string type: 
        :param string event: 
        :param ProgressUpdateEventBody body: 
        :param integer seq: Sequence number (also known as message ID). For protocol messages of type 'request' this ID can be used to cancel the request.
        """
        self.type = 'event'
        self.event = 'progressUpdate'
        if body is None:
            self.body = ProgressUpdateEventBody()
        else:
            self.body = ProgressUpdateEventBody(update_ids_from_dap=update_ids_from_dap, **body) if body.__class__ != ProgressUpdateEventBody else body
        self.seq = seq
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        type = self.type
        event = self.event
        body = self.body
        seq = self.seq
        dct = {'type': type, 'event': event, 'body': body.to_dict(update_ids_to_dap=update_ids_to_dap), 'seq': seq}
        dct.update(self.kwargs)
        return dct