from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register_event('stopped')
@register
class StoppedEvent(BaseSchema):
    """
    The event indicates that the execution of the debuggee has stopped due to some condition.
    
    This can be caused by a break point previously set, a stepping request has completed, by executing a
    debugger statement etc.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'seq': {'type': 'integer', 'description': "Sequence number (also known as message ID). For protocol messages of type 'request' this ID can be used to cancel the request."}, 'type': {'type': 'string', 'enum': ['event']}, 'event': {'type': 'string', 'enum': ['stopped']}, 'body': {'type': 'object', 'properties': {'reason': {'type': 'string', 'description': "The reason for the event.\nFor backward compatibility this string is shown in the UI if the 'description' attribute is missing (but it must not be translated).", '_enum': ['step', 'breakpoint', 'exception', 'pause', 'entry', 'goto', 'function breakpoint', 'data breakpoint', 'instruction breakpoint']}, 'description': {'type': 'string', 'description': "The full reason for the event, e.g. 'Paused on exception'. This string is shown in the UI as is and must be translated."}, 'threadId': {'type': 'integer', 'description': 'The thread which was stopped.'}, 'preserveFocusHint': {'type': 'boolean', 'description': 'A value of true hints to the frontend that this event should not change the focus.'}, 'text': {'type': 'string', 'description': "Additional information. E.g. if reason is 'exception', text contains the exception name. This string is shown in the UI."}, 'allThreadsStopped': {'type': 'boolean', 'description': "If 'allThreadsStopped' is true, a debug adapter can announce that all threads have stopped.\n- The client should use this information to enable that all threads can be expanded to access their stacktraces.\n- If the attribute is missing or false, only the thread with the given threadId can be expanded."}, 'hitBreakpointIds': {'type': 'array', 'items': {'type': 'integer'}, 'description': 'Ids of the breakpoints that triggered the event. In most cases there will be only a single breakpoint but here are some examples for multiple breakpoints:\n- Different types of breakpoints map to the same location.\n- Multiple source breakpoints get collapsed to the same instruction by the compiler/runtime.\n- Multiple function breakpoints with different function names map to the same location.'}}, 'required': ['reason']}}
    __refs__ = set(['body'])
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, body, seq=-1, update_ids_from_dap=False, **kwargs):
        """
        :param string type: 
        :param string event: 
        :param StoppedEventBody body: 
        :param integer seq: Sequence number (also known as message ID). For protocol messages of type 'request' this ID can be used to cancel the request.
        """
        self.type = 'event'
        self.event = 'stopped'
        if body is None:
            self.body = StoppedEventBody()
        else:
            self.body = StoppedEventBody(update_ids_from_dap=update_ids_from_dap, **body) if body.__class__ != StoppedEventBody else body
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