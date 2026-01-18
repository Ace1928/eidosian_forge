from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class ThreadsResponseBody(BaseSchema):
    """
    "body" of ThreadsResponse

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'threads': {'type': 'array', 'items': {'$ref': '#/definitions/Thread'}, 'description': 'All threads.'}}
    __refs__ = set()
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, threads, update_ids_from_dap=False, **kwargs):
        """
        :param array threads: All threads.
        """
        self.threads = threads
        if update_ids_from_dap and self.threads:
            for o in self.threads:
                Thread.update_dict_ids_from_dap(o)
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        threads = self.threads
        if threads and hasattr(threads[0], 'to_dict'):
            threads = [x.to_dict() for x in threads]
        dct = {'threads': [Thread.update_dict_ids_to_dap(o) for o in threads] if update_ids_to_dap and threads else threads}
        dct.update(self.kwargs)
        return dct