from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class PauseArguments(BaseSchema):
    """
    Arguments for 'pause' request.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'threadId': {'type': 'integer', 'description': 'Pause execution for this thread.'}}
    __refs__ = set()
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, threadId, update_ids_from_dap=False, **kwargs):
        """
        :param integer threadId: Pause execution for this thread.
        """
        self.threadId = threadId
        if update_ids_from_dap:
            self.threadId = self._translate_id_from_dap(self.threadId)
        self.kwargs = kwargs

    @classmethod
    def update_dict_ids_from_dap(cls, dct):
        if 'threadId' in dct:
            dct['threadId'] = cls._translate_id_from_dap(dct['threadId'])
        return dct

    def to_dict(self, update_ids_to_dap=False):
        threadId = self.threadId
        if update_ids_to_dap:
            if threadId is not None:
                threadId = self._translate_id_to_dap(threadId)
        dct = {'threadId': threadId}
        dct.update(self.kwargs)
        return dct

    @classmethod
    def update_dict_ids_to_dap(cls, dct):
        if 'threadId' in dct:
            dct['threadId'] = cls._translate_id_to_dap(dct['threadId'])
        return dct