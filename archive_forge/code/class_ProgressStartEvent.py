from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register_event('progressStart')
@register
class ProgressStartEvent(BaseSchema):
    """
    The event signals that a long running operation is about to start and
    
    provides additional information for the client to set up a corresponding progress and cancellation
    UI.
    
    The client is free to delay the showing of the UI in order to reduce flicker.
    
    This event should only be sent if the client has passed the value true for the
    'supportsProgressReporting' capability of the 'initialize' request.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'seq': {'type': 'integer', 'description': "Sequence number (also known as message ID). For protocol messages of type 'request' this ID can be used to cancel the request."}, 'type': {'type': 'string', 'enum': ['event']}, 'event': {'type': 'string', 'enum': ['progressStart']}, 'body': {'type': 'object', 'properties': {'progressId': {'type': 'string', 'description': "An ID that must be used in subsequent 'progressUpdate' and 'progressEnd' events to make them refer to the same progress reporting.\nIDs must be unique within a debug session."}, 'title': {'type': 'string', 'description': 'Mandatory (short) title of the progress reporting. Shown in the UI to describe the long running operation.'}, 'requestId': {'type': 'integer', 'description': 'The request ID that this progress report is related to. If specified a debug adapter is expected to emit\nprogress events for the long running request until the request has been either completed or cancelled.\nIf the request ID is omitted, the progress report is assumed to be related to some general activity of the debug adapter.'}, 'cancellable': {'type': 'boolean', 'description': "If true, the request that reports progress may be canceled with a 'cancel' request.\nSo this property basically controls whether the client should use UX that supports cancellation.\nClients that don't support cancellation are allowed to ignore the setting."}, 'message': {'type': 'string', 'description': 'Optional, more detailed progress message.'}, 'percentage': {'type': 'number', 'description': 'Optional progress percentage to display (value range: 0 to 100). If omitted no percentage will be shown.'}}, 'required': ['progressId', 'title']}}
    __refs__ = set(['body'])
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, body, seq=-1, update_ids_from_dap=False, **kwargs):
        """
        :param string type: 
        :param string event: 
        :param ProgressStartEventBody body: 
        :param integer seq: Sequence number (also known as message ID). For protocol messages of type 'request' this ID can be used to cancel the request.
        """
        self.type = 'event'
        self.event = 'progressStart'
        if body is None:
            self.body = ProgressStartEventBody()
        else:
            self.body = ProgressStartEventBody(update_ids_from_dap=update_ids_from_dap, **body) if body.__class__ != ProgressStartEventBody else body
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