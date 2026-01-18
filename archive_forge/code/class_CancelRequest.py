from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register_request('cancel')
@register
class CancelRequest(BaseSchema):
    """
    The 'cancel' request is used by the frontend in two situations:
    
    - to indicate that it is no longer interested in the result produced by a specific request issued
    earlier
    
    - to cancel a progress sequence. Clients should only call this request if the capability
    'supportsCancelRequest' is true.
    
    This request has a hint characteristic: a debug adapter can only be expected to make a 'best effort'
    in honouring this request but there are no guarantees.
    
    The 'cancel' request may return an error if it could not cancel an operation but a frontend should
    refrain from presenting this error to end users.
    
    A frontend client should only call this request if the capability 'supportsCancelRequest' is true.
    
    The request that got canceled still needs to send a response back. This can either be a normal
    result ('success' attribute true)
    
    or an error response ('success' attribute false and the 'message' set to 'cancelled').
    
    Returning partial results from a cancelled request is possible but please note that a frontend
    client has no generic way for detecting that a response is partial or not.
    
    The progress that got cancelled still needs to send a 'progressEnd' event back.
    
    A client should not assume that progress just got cancelled after sending the 'cancel' request.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'seq': {'type': 'integer', 'description': "Sequence number (also known as message ID). For protocol messages of type 'request' this ID can be used to cancel the request."}, 'type': {'type': 'string', 'enum': ['request']}, 'command': {'type': 'string', 'enum': ['cancel']}, 'arguments': {'type': 'CancelArguments'}}
    __refs__ = set(['arguments'])
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, seq=-1, arguments=None, update_ids_from_dap=False, **kwargs):
        """
        :param string type: 
        :param string command: 
        :param integer seq: Sequence number (also known as message ID). For protocol messages of type 'request' this ID can be used to cancel the request.
        :param CancelArguments arguments: 
        """
        self.type = 'request'
        self.command = 'cancel'
        self.seq = seq
        if arguments is None:
            self.arguments = CancelArguments()
        else:
            self.arguments = CancelArguments(update_ids_from_dap=update_ids_from_dap, **arguments) if arguments.__class__ != CancelArguments else arguments
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        type = self.type
        command = self.command
        seq = self.seq
        arguments = self.arguments
        dct = {'type': type, 'command': command, 'seq': seq}
        if arguments is not None:
            dct['arguments'] = arguments.to_dict(update_ids_to_dap=update_ids_to_dap)
        dct.update(self.kwargs)
        return dct