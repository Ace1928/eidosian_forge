from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register_request('stackTrace')
@register
class StackTraceRequest(BaseSchema):
    """
    The request returns a stacktrace from the current execution state of a given thread.
    
    A client can request all stack frames by omitting the startFrame and levels arguments. For
    performance conscious clients and if the debug adapter's 'supportsDelayedStackTraceLoading'
    capability is true, stack frames can be retrieved in a piecemeal way with the startFrame and levels
    arguments. The response of the stackTrace request may contain a totalFrames property that hints at
    the total number of frames in the stack. If a client needs this total number upfront, it can issue a
    request for a single (first) frame and depending on the value of totalFrames decide how to proceed.
    In any case a client should be prepared to receive less frames than requested, which is an
    indication that the end of the stack has been reached.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'seq': {'type': 'integer', 'description': "Sequence number (also known as message ID). For protocol messages of type 'request' this ID can be used to cancel the request."}, 'type': {'type': 'string', 'enum': ['request']}, 'command': {'type': 'string', 'enum': ['stackTrace']}, 'arguments': {'type': 'StackTraceArguments'}}
    __refs__ = set(['arguments'])
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, arguments, seq=-1, update_ids_from_dap=False, **kwargs):
        """
        :param string type: 
        :param string command: 
        :param StackTraceArguments arguments: 
        :param integer seq: Sequence number (also known as message ID). For protocol messages of type 'request' this ID can be used to cancel the request.
        """
        self.type = 'request'
        self.command = 'stackTrace'
        if arguments is None:
            self.arguments = StackTraceArguments()
        else:
            self.arguments = StackTraceArguments(update_ids_from_dap=update_ids_from_dap, **arguments) if arguments.__class__ != StackTraceArguments else arguments
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