from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register_request('stepInTargets')
@register
class StepInTargetsRequest(BaseSchema):
    """
    This request retrieves the possible stepIn targets for the specified stack frame.
    
    These targets can be used in the 'stepIn' request.
    
    The StepInTargets may only be called if the 'supportsStepInTargetsRequest' capability exists and is
    true.
    
    Clients should only call this request if the capability 'supportsStepInTargetsRequest' is true.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'seq': {'type': 'integer', 'description': "Sequence number (also known as message ID). For protocol messages of type 'request' this ID can be used to cancel the request."}, 'type': {'type': 'string', 'enum': ['request']}, 'command': {'type': 'string', 'enum': ['stepInTargets']}, 'arguments': {'type': 'StepInTargetsArguments'}}
    __refs__ = set(['arguments'])
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, arguments, seq=-1, update_ids_from_dap=False, **kwargs):
        """
        :param string type: 
        :param string command: 
        :param StepInTargetsArguments arguments: 
        :param integer seq: Sequence number (also known as message ID). For protocol messages of type 'request' this ID can be used to cancel the request.
        """
        self.type = 'request'
        self.command = 'stepInTargets'
        if arguments is None:
            self.arguments = StepInTargetsArguments()
        else:
            self.arguments = StepInTargetsArguments(update_ids_from_dap=update_ids_from_dap, **arguments) if arguments.__class__ != StepInTargetsArguments else arguments
        self.seq = seq
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        type = self.type
        command = self.command
        arguments = self.arguments
        seq = self.seq
        dct = {'type': type, 'command': command, 'arguments': arguments.to_dict(update_ids_to_dap=update_ids_to_dap), 'seq': seq}
        dct.update(self.kwargs)
        return dct