from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register_request('stepIn')
@register
class StepInRequest(BaseSchema):
    """
    The request resumes the given thread to step into a function/method and allows all other threads to
    run freely by resuming them.
    
    If the debug adapter supports single thread execution (see capability
    'supportsSingleThreadExecutionRequests') setting the 'singleThread' argument to true prevents other
    suspended threads from resuming.
    
    If the request cannot step into a target, 'stepIn' behaves like the 'next' request.
    
    The debug adapter first sends the response and then a 'stopped' event (with reason 'step') after the
    step has completed.
    
    If there are multiple function/method calls (or other targets) on the source line,
    
    the optional argument 'targetId' can be used to control into which target the 'stepIn' should occur.
    
    The list of possible targets for a given source line can be retrieved via the 'stepInTargets'
    request.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'seq': {'type': 'integer', 'description': "Sequence number (also known as message ID). For protocol messages of type 'request' this ID can be used to cancel the request."}, 'type': {'type': 'string', 'enum': ['request']}, 'command': {'type': 'string', 'enum': ['stepIn']}, 'arguments': {'type': 'StepInArguments'}}
    __refs__ = set(['arguments'])
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, arguments, seq=-1, update_ids_from_dap=False, **kwargs):
        """
        :param string type: 
        :param string command: 
        :param StepInArguments arguments: 
        :param integer seq: Sequence number (also known as message ID). For protocol messages of type 'request' this ID can be used to cancel the request.
        """
        self.type = 'request'
        self.command = 'stepIn'
        if arguments is None:
            self.arguments = StepInArguments()
        else:
            self.arguments = StepInArguments(update_ids_from_dap=update_ids_from_dap, **arguments) if arguments.__class__ != StepInArguments else arguments
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