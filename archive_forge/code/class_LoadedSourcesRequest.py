from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register_request('loadedSources')
@register
class LoadedSourcesRequest(BaseSchema):
    """
    Retrieves the set of all sources currently loaded by the debugged process.
    
    Clients should only call this request if the capability 'supportsLoadedSourcesRequest' is true.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'seq': {'type': 'integer', 'description': "Sequence number (also known as message ID). For protocol messages of type 'request' this ID can be used to cancel the request."}, 'type': {'type': 'string', 'enum': ['request']}, 'command': {'type': 'string', 'enum': ['loadedSources']}, 'arguments': {'type': 'LoadedSourcesArguments'}}
    __refs__ = set(['arguments'])
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, seq=-1, arguments=None, update_ids_from_dap=False, **kwargs):
        """
        :param string type: 
        :param string command: 
        :param integer seq: Sequence number (also known as message ID). For protocol messages of type 'request' this ID can be used to cancel the request.
        :param LoadedSourcesArguments arguments: 
        """
        self.type = 'request'
        self.command = 'loadedSources'
        self.seq = seq
        if arguments is None:
            self.arguments = LoadedSourcesArguments()
        else:
            self.arguments = LoadedSourcesArguments(update_ids_from_dap=update_ids_from_dap, **arguments) if arguments.__class__ != LoadedSourcesArguments else arguments
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