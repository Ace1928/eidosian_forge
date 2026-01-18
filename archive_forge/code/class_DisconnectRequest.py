from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register_request('disconnect')
@register
class DisconnectRequest(BaseSchema):
    """
    The 'disconnect' request is sent from the client to the debug adapter in order to stop debugging.
    
    It asks the debug adapter to disconnect from the debuggee and to terminate the debug adapter.
    
    If the debuggee has been started with the 'launch' request, the 'disconnect' request terminates the
    debuggee.
    
    If the 'attach' request was used to connect to the debuggee, 'disconnect' does not terminate the
    debuggee.
    
    This behavior can be controlled with the 'terminateDebuggee' argument (if supported by the debug
    adapter).

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'seq': {'type': 'integer', 'description': "Sequence number (also known as message ID). For protocol messages of type 'request' this ID can be used to cancel the request."}, 'type': {'type': 'string', 'enum': ['request']}, 'command': {'type': 'string', 'enum': ['disconnect']}, 'arguments': {'type': 'DisconnectArguments'}}
    __refs__ = set(['arguments'])
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, seq=-1, arguments=None, update_ids_from_dap=False, **kwargs):
        """
        :param string type: 
        :param string command: 
        :param integer seq: Sequence number (also known as message ID). For protocol messages of type 'request' this ID can be used to cancel the request.
        :param DisconnectArguments arguments: 
        """
        self.type = 'request'
        self.command = 'disconnect'
        self.seq = seq
        if arguments is None:
            self.arguments = DisconnectArguments()
        else:
            self.arguments = DisconnectArguments(update_ids_from_dap=update_ids_from_dap, **arguments) if arguments.__class__ != DisconnectArguments else arguments
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