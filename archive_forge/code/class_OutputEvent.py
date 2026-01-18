from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register_event('output')
@register
class OutputEvent(BaseSchema):
    """
    The event indicates that the target has produced some output.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'seq': {'type': 'integer', 'description': "Sequence number (also known as message ID). For protocol messages of type 'request' this ID can be used to cancel the request."}, 'type': {'type': 'string', 'enum': ['event']}, 'event': {'type': 'string', 'enum': ['output']}, 'body': {'type': 'object', 'properties': {'category': {'type': 'string', 'description': "The output category. If not specified or if the category is not understand by the client, 'console' is assumed.", '_enum': ['console', 'important', 'stdout', 'stderr', 'telemetry'], 'enumDescriptions': ["Show the output in the client's default message UI, e.g. a 'debug console'. This category should only be used for informational output from the debugger (as opposed to the debuggee).", "A hint for the client to show the ouput in the client's UI for important and highly visible information, e.g. as a popup notification. This category should only be used for important messages from the debugger (as opposed to the debuggee). Since this category value is a hint, clients might ignore the hint and assume the 'console' category.", 'Show the output as normal program output from the debuggee.', 'Show the output as error program output from the debuggee.', 'Send the output to telemetry instead of showing it to the user.']}, 'output': {'type': 'string', 'description': 'The output to report.'}, 'group': {'type': 'string', 'description': 'Support for keeping an output log organized by grouping related messages.', 'enum': ['start', 'startCollapsed', 'end'], 'enumDescriptions': ["Start a new group in expanded mode. Subsequent output events are members of the group and should be shown indented.\nThe 'output' attribute becomes the name of the group and is not indented.", "Start a new group in collapsed mode. Subsequent output events are members of the group and should be shown indented (as soon as the group is expanded).\nThe 'output' attribute becomes the name of the group and is not indented.", "End the current group and decreases the indentation of subsequent output events.\nA non empty 'output' attribute is shown as the unindented end of the group."]}, 'variablesReference': {'type': 'integer', 'description': "If an attribute 'variablesReference' exists and its value is > 0, the output contains objects which can be retrieved by passing 'variablesReference' to the 'variables' request. The value should be less than or equal to 2147483647 (2^31-1)."}, 'source': {'$ref': '#/definitions/Source', 'description': 'An optional source location where the output was produced.'}, 'line': {'type': 'integer', 'description': 'An optional source location line where the output was produced.'}, 'column': {'type': 'integer', 'description': 'An optional source location column where the output was produced.'}, 'data': {'type': ['array', 'boolean', 'integer', 'null', 'number', 'object', 'string'], 'description': "Optional data to report. For the 'telemetry' category the data will be sent to telemetry, for the other categories the data is shown in JSON format."}}, 'required': ['output']}}
    __refs__ = set(['body'])
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, body, seq=-1, update_ids_from_dap=False, **kwargs):
        """
        :param string type: 
        :param string event: 
        :param OutputEventBody body: 
        :param integer seq: Sequence number (also known as message ID). For protocol messages of type 'request' this ID can be used to cancel the request.
        """
        self.type = 'event'
        self.event = 'output'
        if body is None:
            self.body = OutputEventBody()
        else:
            self.body = OutputEventBody(update_ids_from_dap=update_ids_from_dap, **body) if body.__class__ != OutputEventBody else body
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