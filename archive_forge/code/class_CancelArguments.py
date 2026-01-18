from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class CancelArguments(BaseSchema):
    """
    Arguments for 'cancel' request.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'requestId': {'type': 'integer', 'description': "The ID (attribute 'seq') of the request to cancel. If missing no request is cancelled.\nBoth a 'requestId' and a 'progressId' can be specified in one request."}, 'progressId': {'type': 'string', 'description': "The ID (attribute 'progressId') of the progress to cancel. If missing no progress is cancelled.\nBoth a 'requestId' and a 'progressId' can be specified in one request."}}
    __refs__ = set()
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, requestId=None, progressId=None, update_ids_from_dap=False, **kwargs):
        """
        :param integer requestId: The ID (attribute 'seq') of the request to cancel. If missing no request is cancelled.
        Both a 'requestId' and a 'progressId' can be specified in one request.
        :param string progressId: The ID (attribute 'progressId') of the progress to cancel. If missing no progress is cancelled.
        Both a 'requestId' and a 'progressId' can be specified in one request.
        """
        self.requestId = requestId
        self.progressId = progressId
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        requestId = self.requestId
        progressId = self.progressId
        dct = {}
        if requestId is not None:
            dct['requestId'] = requestId
        if progressId is not None:
            dct['progressId'] = progressId
        dct.update(self.kwargs)
        return dct