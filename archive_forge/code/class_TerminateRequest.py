from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register_request('terminate')
@register
class TerminateRequest(BaseSchema):
    """
    The 'terminate' request is sent from the client to the debug adapter in order to give the debuggee a
    chance for terminating itself.
    
    Clients should only call this request if the capability 'supportsTerminateRequest' is true.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'seq': {'type': 'integer', 'description': "Sequence number (also known as message ID). For protocol messages of type 'request' this ID can be used to cancel the request."}, 'type': {'type': 'string', 'enum': ['request']}, 'command': {'type': 'string', 'enum': ['terminate']}, 'arguments': {'type': 'TerminateArguments'}}
    __refs__ = set(['arguments'])
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, seq=-1, arguments=None, update_ids_from_dap=False, **kwargs):
        """
        :param string type: 
        :param string command: 
        :param integer seq: Sequence number (also known as message ID). For protocol messages of type 'request' this ID can be used to cancel the request.
        :param TerminateArguments arguments: 
        """
        self.type = 'request'
        self.command = 'terminate'
        self.seq = seq
        if arguments is None:
            self.arguments = TerminateArguments()
        else:
            self.arguments = TerminateArguments(update_ids_from_dap=update_ids_from_dap, **arguments) if arguments.__class__ != TerminateArguments else arguments
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