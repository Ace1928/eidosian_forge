from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register_request('evaluate')
@register
class EvaluateRequest(BaseSchema):
    """
    Evaluates the given expression in the context of the top most stack frame.
    
    The expression has access to any variables and arguments that are in scope.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'seq': {'type': 'integer', 'description': "Sequence number (also known as message ID). For protocol messages of type 'request' this ID can be used to cancel the request."}, 'type': {'type': 'string', 'enum': ['request']}, 'command': {'type': 'string', 'enum': ['evaluate']}, 'arguments': {'type': 'EvaluateArguments'}}
    __refs__ = set(['arguments'])
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, arguments, seq=-1, update_ids_from_dap=False, **kwargs):
        """
        :param string type: 
        :param string command: 
        :param EvaluateArguments arguments: 
        :param integer seq: Sequence number (also known as message ID). For protocol messages of type 'request' this ID can be used to cancel the request.
        """
        self.type = 'request'
        self.command = 'evaluate'
        if arguments is None:
            self.arguments = EvaluateArguments()
        else:
            self.arguments = EvaluateArguments(update_ids_from_dap=update_ids_from_dap, **arguments) if arguments.__class__ != EvaluateArguments else arguments
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