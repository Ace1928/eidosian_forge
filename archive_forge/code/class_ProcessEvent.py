from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register_event('process')
@register
class ProcessEvent(BaseSchema):
    """
    The event indicates that the debugger has begun debugging a new process. Either one that it has
    launched, or one that it has attached to.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'seq': {'type': 'integer', 'description': "Sequence number (also known as message ID). For protocol messages of type 'request' this ID can be used to cancel the request."}, 'type': {'type': 'string', 'enum': ['event']}, 'event': {'type': 'string', 'enum': ['process']}, 'body': {'type': 'object', 'properties': {'name': {'type': 'string', 'description': "The logical name of the process. This is usually the full path to process's executable file. Example: /home/example/myproj/program.js."}, 'systemProcessId': {'type': 'integer', 'description': 'The system process id of the debugged process. This property will be missing for non-system processes.'}, 'isLocalProcess': {'type': 'boolean', 'description': 'If true, the process is running on the same computer as the debug adapter.'}, 'startMethod': {'type': 'string', 'enum': ['launch', 'attach', 'attachForSuspendedLaunch'], 'description': 'Describes how the debug engine started debugging this process.', 'enumDescriptions': ['Process was launched under the debugger.', 'Debugger attached to an existing process.', 'A project launcher component has launched a new process in a suspended state and then asked the debugger to attach.']}, 'pointerSize': {'type': 'integer', 'description': 'The size of a pointer or address for this process, in bits. This value may be used by clients when formatting addresses for display.'}}, 'required': ['name']}}
    __refs__ = set(['body'])
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, body, seq=-1, update_ids_from_dap=False, **kwargs):
        """
        :param string type: 
        :param string event: 
        :param ProcessEventBody body: 
        :param integer seq: Sequence number (also known as message ID). For protocol messages of type 'request' this ID can be used to cancel the request.
        """
        self.type = 'event'
        self.event = 'process'
        if body is None:
            self.body = ProcessEventBody()
        else:
            self.body = ProcessEventBody(update_ids_from_dap=update_ids_from_dap, **body) if body.__class__ != ProcessEventBody else body
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