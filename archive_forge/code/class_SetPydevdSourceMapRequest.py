from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register_request('setPydevdSourceMap')
@register
class SetPydevdSourceMapRequest(BaseSchema):
    """
    Sets multiple PydevdSourceMap for a single source and clears all previous PydevdSourceMap in that
    source.
    
    i.e.: Maps paths and lines in a 1:N mapping (use case: map a single file in the IDE to multiple
    IPython cells).
    
    To clear all PydevdSourceMap for a source, specify an empty array.
    
    Interaction with breakpoints: When a new mapping is sent, breakpoints that match the source (or
    previously matched a source) are reapplied.
    
    Interaction with launch pathMapping: both mappings are independent. This mapping is applied after
    the launch pathMapping.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'seq': {'type': 'integer', 'description': "Sequence number (also known as message ID). For protocol messages of type 'request' this ID can be used to cancel the request."}, 'type': {'type': 'string', 'enum': ['request']}, 'command': {'type': 'string', 'enum': ['setPydevdSourceMap']}, 'arguments': {'type': 'SetPydevdSourceMapArguments'}}
    __refs__ = set(['arguments'])
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, arguments, seq=-1, update_ids_from_dap=False, **kwargs):
        """
        :param string type: 
        :param string command: 
        :param SetPydevdSourceMapArguments arguments: 
        :param integer seq: Sequence number (also known as message ID). For protocol messages of type 'request' this ID can be used to cancel the request.
        """
        self.type = 'request'
        self.command = 'setPydevdSourceMap'
        if arguments is None:
            self.arguments = SetPydevdSourceMapArguments()
        else:
            self.arguments = SetPydevdSourceMapArguments(update_ids_from_dap=update_ids_from_dap, **arguments) if arguments.__class__ != SetPydevdSourceMapArguments else arguments
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