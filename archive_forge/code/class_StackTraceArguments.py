from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class StackTraceArguments(BaseSchema):
    """
    Arguments for 'stackTrace' request.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'threadId': {'type': 'integer', 'description': 'Retrieve the stacktrace for this thread.'}, 'startFrame': {'type': 'integer', 'description': 'The index of the first frame to return; if omitted frames start at 0.'}, 'levels': {'type': 'integer', 'description': 'The maximum number of frames to return. If levels is not specified or 0, all frames are returned.'}, 'format': {'description': "Specifies details on how to format the stack frames.\nThe attribute is only honored by a debug adapter if the capability 'supportsValueFormattingOptions' is true.", 'type': 'StackFrameFormat'}}
    __refs__ = set(['format'])
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, threadId, startFrame=None, levels=None, format=None, update_ids_from_dap=False, **kwargs):
        """
        :param integer threadId: Retrieve the stacktrace for this thread.
        :param integer startFrame: The index of the first frame to return; if omitted frames start at 0.
        :param integer levels: The maximum number of frames to return. If levels is not specified or 0, all frames are returned.
        :param StackFrameFormat format: Specifies details on how to format the stack frames.
        The attribute is only honored by a debug adapter if the capability 'supportsValueFormattingOptions' is true.
        """
        self.threadId = threadId
        self.startFrame = startFrame
        self.levels = levels
        if format is None:
            self.format = StackFrameFormat()
        else:
            self.format = StackFrameFormat(update_ids_from_dap=update_ids_from_dap, **format) if format.__class__ != StackFrameFormat else format
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
        startFrame = self.startFrame
        levels = self.levels
        format = self.format
        if update_ids_to_dap:
            if threadId is not None:
                threadId = self._translate_id_to_dap(threadId)
        dct = {'threadId': threadId}
        if startFrame is not None:
            dct['startFrame'] = startFrame
        if levels is not None:
            dct['levels'] = levels
        if format is not None:
            dct['format'] = format.to_dict(update_ids_to_dap=update_ids_to_dap)
        dct.update(self.kwargs)
        return dct

    @classmethod
    def update_dict_ids_to_dap(cls, dct):
        if 'threadId' in dct:
            dct['threadId'] = cls._translate_id_to_dap(dct['threadId'])
        return dct