from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class TerminateThreadsArguments(BaseSchema):
    """
    Arguments for 'terminateThreads' request.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'threadIds': {'type': 'array', 'items': {'type': 'integer'}, 'description': 'Ids of threads to be terminated.'}}
    __refs__ = set()
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, threadIds=None, update_ids_from_dap=False, **kwargs):
        """
        :param array threadIds: Ids of threads to be terminated.
        """
        self.threadIds = threadIds
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        threadIds = self.threadIds
        if threadIds and hasattr(threadIds[0], 'to_dict'):
            threadIds = [x.to_dict() for x in threadIds]
        dct = {}
        if threadIds is not None:
            dct['threadIds'] = threadIds
        dct.update(self.kwargs)
        return dct