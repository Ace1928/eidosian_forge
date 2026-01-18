from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register_event('progressEnd')
@register
class ProgressEndEvent(BaseSchema):
    """
    The event signals the end of the progress reporting with an optional final message.
    
    This event should only be sent if the client has passed the value true for the
    'supportsProgressReporting' capability of the 'initialize' request.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'seq': {'type': 'integer', 'description': "Sequence number (also known as message ID). For protocol messages of type 'request' this ID can be used to cancel the request."}, 'type': {'type': 'string', 'enum': ['event']}, 'event': {'type': 'string', 'enum': ['progressEnd']}, 'body': {'type': 'object', 'properties': {'progressId': {'type': 'string', 'description': "The ID that was introduced in the initial 'ProgressStartEvent'."}, 'message': {'type': 'string', 'description': 'Optional, more detailed progress message. If omitted, the previous message (if any) is used.'}}, 'required': ['progressId']}}
    __refs__ = set(['body'])
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, body, seq=-1, update_ids_from_dap=False, **kwargs):
        """
        :param string type: 
        :param string event: 
        :param ProgressEndEventBody body: 
        :param integer seq: Sequence number (also known as message ID). For protocol messages of type 'request' this ID can be used to cancel the request.
        """
        self.type = 'event'
        self.event = 'progressEnd'
        if body is None:
            self.body = ProgressEndEventBody()
        else:
            self.body = ProgressEndEventBody(update_ids_from_dap=update_ids_from_dap, **body) if body.__class__ != ProgressEndEventBody else body
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