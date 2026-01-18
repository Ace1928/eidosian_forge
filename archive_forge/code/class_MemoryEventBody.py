from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class MemoryEventBody(BaseSchema):
    """
    "body" of MemoryEvent

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'memoryReference': {'type': 'string', 'description': 'Memory reference of a memory range that has been updated.'}, 'offset': {'type': 'integer', 'description': 'Starting offset in bytes where memory has been updated. Can be negative.'}, 'count': {'type': 'integer', 'description': 'Number of bytes updated.'}}
    __refs__ = set()
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, memoryReference, offset, count, update_ids_from_dap=False, **kwargs):
        """
        :param string memoryReference: Memory reference of a memory range that has been updated.
        :param integer offset: Starting offset in bytes where memory has been updated. Can be negative.
        :param integer count: Number of bytes updated.
        """
        self.memoryReference = memoryReference
        self.offset = offset
        self.count = count
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        memoryReference = self.memoryReference
        offset = self.offset
        count = self.count
        dct = {'memoryReference': memoryReference, 'offset': offset, 'count': count}
        dct.update(self.kwargs)
        return dct