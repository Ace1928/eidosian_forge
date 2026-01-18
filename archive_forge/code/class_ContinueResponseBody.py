from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class ContinueResponseBody(BaseSchema):
    """
    "body" of ContinueResponse

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'allThreadsContinued': {'type': 'boolean', 'description': 'The value true (or a missing property) signals to the client that all threads have been resumed. The value false must be returned if not all threads were resumed.'}}
    __refs__ = set()
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, allThreadsContinued=None, update_ids_from_dap=False, **kwargs):
        """
        :param boolean allThreadsContinued: The value true (or a missing property) signals to the client that all threads have been resumed. The value false must be returned if not all threads were resumed.
        """
        self.allThreadsContinued = allThreadsContinued
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        allThreadsContinued = self.allThreadsContinued
        dct = {}
        if allThreadsContinued is not None:
            dct['allThreadsContinued'] = allThreadsContinued
        dct.update(self.kwargs)
        return dct