from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class ProgressUpdateEventBody(BaseSchema):
    """
    "body" of ProgressUpdateEvent

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'progressId': {'type': 'string', 'description': "The ID that was introduced in the initial 'progressStart' event."}, 'message': {'type': 'string', 'description': 'Optional, more detailed progress message. If omitted, the previous message (if any) is used.'}, 'percentage': {'type': 'number', 'description': 'Optional progress percentage to display (value range: 0 to 100). If omitted no percentage will be shown.'}}
    __refs__ = set()
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, progressId, message=None, percentage=None, update_ids_from_dap=False, **kwargs):
        """
        :param string progressId: The ID that was introduced in the initial 'progressStart' event.
        :param string message: Optional, more detailed progress message. If omitted, the previous message (if any) is used.
        :param number percentage: Optional progress percentage to display (value range: 0 to 100). If omitted no percentage will be shown.
        """
        self.progressId = progressId
        self.message = message
        self.percentage = percentage
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        progressId = self.progressId
        message = self.message
        percentage = self.percentage
        dct = {'progressId': progressId}
        if message is not None:
            dct['message'] = message
        if percentage is not None:
            dct['percentage'] = percentage
        dct.update(self.kwargs)
        return dct