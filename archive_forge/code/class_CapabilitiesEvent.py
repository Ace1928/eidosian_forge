from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register_event('capabilities')
@register
class CapabilitiesEvent(BaseSchema):
    """
    The event indicates that one or more capabilities have changed.
    
    Since the capabilities are dependent on the frontend and its UI, it might not be possible to change
    that at random times (or too late).
    
    Consequently this event has a hint characteristic: a frontend can only be expected to make a 'best
    effort' in honouring individual capabilities but there are no guarantees.
    
    Only changed capabilities need to be included, all other capabilities keep their values.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'seq': {'type': 'integer', 'description': "Sequence number (also known as message ID). For protocol messages of type 'request' this ID can be used to cancel the request."}, 'type': {'type': 'string', 'enum': ['event']}, 'event': {'type': 'string', 'enum': ['capabilities']}, 'body': {'type': 'object', 'properties': {'capabilities': {'$ref': '#/definitions/Capabilities', 'description': 'The set of updated capabilities.'}}, 'required': ['capabilities']}}
    __refs__ = set(['body'])
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, body, seq=-1, update_ids_from_dap=False, **kwargs):
        """
        :param string type: 
        :param string event: 
        :param CapabilitiesEventBody body: 
        :param integer seq: Sequence number (also known as message ID). For protocol messages of type 'request' this ID can be used to cancel the request.
        """
        self.type = 'event'
        self.event = 'capabilities'
        if body is None:
            self.body = CapabilitiesEventBody()
        else:
            self.body = CapabilitiesEventBody(update_ids_from_dap=update_ids_from_dap, **body) if body.__class__ != CapabilitiesEventBody else body
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