from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register_request('continue')
@register
class ContinueRequest(BaseSchema):
    """
    The request resumes execution of all threads. If the debug adapter supports single thread execution
    (see capability 'supportsSingleThreadExecutionRequests') setting the 'singleThread' argument to true
    resumes only the specified thread. If not all threads were resumed, the 'allThreadsContinued'
    attribute of the response must be set to false.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'seq': {'type': 'integer', 'description': "Sequence number (also known as message ID). For protocol messages of type 'request' this ID can be used to cancel the request."}, 'type': {'type': 'string', 'enum': ['request']}, 'command': {'type': 'string', 'enum': ['continue']}, 'arguments': {'type': 'ContinueArguments'}}
    __refs__ = set(['arguments'])
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, arguments, seq=-1, update_ids_from_dap=False, **kwargs):
        """
        :param string type: 
        :param string command: 
        :param ContinueArguments arguments: 
        :param integer seq: Sequence number (also known as message ID). For protocol messages of type 'request' this ID can be used to cancel the request.
        """
        self.type = 'request'
        self.command = 'continue'
        if arguments is None:
            self.arguments = ContinueArguments()
        else:
            self.arguments = ContinueArguments(update_ids_from_dap=update_ids_from_dap, **arguments) if arguments.__class__ != ContinueArguments else arguments
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