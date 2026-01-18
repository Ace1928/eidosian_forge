from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class ProgressEndEventBody(BaseSchema):
    """
    "body" of ProgressEndEvent

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'progressId': {'type': 'string', 'description': "The ID that was introduced in the initial 'ProgressStartEvent'."}, 'message': {'type': 'string', 'description': 'Optional, more detailed progress message. If omitted, the previous message (if any) is used.'}}
    __refs__ = set()
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, progressId, message=None, update_ids_from_dap=False, **kwargs):
        """
        :param string progressId: The ID that was introduced in the initial 'ProgressStartEvent'.
        :param string message: Optional, more detailed progress message. If omitted, the previous message (if any) is used.
        """
        self.progressId = progressId
        self.message = message
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        progressId = self.progressId
        message = self.message
        dct = {'progressId': progressId}
        if message is not None:
            dct['message'] = message
        dct.update(self.kwargs)
        return dct