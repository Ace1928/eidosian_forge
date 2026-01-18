from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class ReadMemoryArguments(BaseSchema):
    """
    Arguments for 'readMemory' request.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'memoryReference': {'type': 'string', 'description': 'Memory reference to the base location from which data should be read.'}, 'offset': {'type': 'integer', 'description': 'Optional offset (in bytes) to be applied to the reference location before reading data. Can be negative.'}, 'count': {'type': 'integer', 'description': 'Number of bytes to read at the specified location and offset.'}}
    __refs__ = set()
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, memoryReference, count, offset=None, update_ids_from_dap=False, **kwargs):
        """
        :param string memoryReference: Memory reference to the base location from which data should be read.
        :param integer count: Number of bytes to read at the specified location and offset.
        :param integer offset: Optional offset (in bytes) to be applied to the reference location before reading data. Can be negative.
        """
        self.memoryReference = memoryReference
        self.count = count
        self.offset = offset
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        memoryReference = self.memoryReference
        count = self.count
        offset = self.offset
        dct = {'memoryReference': memoryReference, 'count': count}
        if offset is not None:
            dct['offset'] = offset
        dct.update(self.kwargs)
        return dct