from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class StepInArguments(BaseSchema):
    """
    Arguments for 'stepIn' request.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'threadId': {'type': 'integer', 'description': 'Specifies the thread for which to resume execution for one step-into (of the given granularity).'}, 'singleThread': {'type': 'boolean', 'description': 'If this optional flag is true, all other suspended threads are not resumed.'}, 'targetId': {'type': 'integer', 'description': 'Optional id of the target to step into.'}, 'granularity': {'description': "Optional granularity to step. If no granularity is specified, a granularity of 'statement' is assumed.", 'type': 'SteppingGranularity'}}
    __refs__ = set(['granularity'])
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, threadId, singleThread=None, targetId=None, granularity=None, update_ids_from_dap=False, **kwargs):
        """
        :param integer threadId: Specifies the thread for which to resume execution for one step-into (of the given granularity).
        :param boolean singleThread: If this optional flag is true, all other suspended threads are not resumed.
        :param integer targetId: Optional id of the target to step into.
        :param SteppingGranularity granularity: Optional granularity to step. If no granularity is specified, a granularity of 'statement' is assumed.
        """
        self.threadId = threadId
        self.singleThread = singleThread
        self.targetId = targetId
        if granularity is not None:
            assert granularity in SteppingGranularity.VALID_VALUES
        self.granularity = granularity
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
        singleThread = self.singleThread
        targetId = self.targetId
        granularity = self.granularity
        if update_ids_to_dap:
            if threadId is not None:
                threadId = self._translate_id_to_dap(threadId)
        dct = {'threadId': threadId}
        if singleThread is not None:
            dct['singleThread'] = singleThread
        if targetId is not None:
            dct['targetId'] = targetId
        if granularity is not None:
            dct['granularity'] = granularity
        dct.update(self.kwargs)
        return dct

    @classmethod
    def update_dict_ids_to_dap(cls, dct):
        if 'threadId' in dct:
            dct['threadId'] = cls._translate_id_to_dap(dct['threadId'])
        return dct