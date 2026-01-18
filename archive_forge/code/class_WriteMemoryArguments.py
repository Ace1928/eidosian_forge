from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class WriteMemoryArguments(BaseSchema):
    """
    Arguments for 'writeMemory' request.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'memoryReference': {'type': 'string', 'description': 'Memory reference to the base location to which data should be written.'}, 'offset': {'type': 'integer', 'description': 'Optional offset (in bytes) to be applied to the reference location before writing data. Can be negative.'}, 'allowPartial': {'type': 'boolean', 'description': "Optional property to control partial writes. If true, the debug adapter should attempt to write memory even if the entire memory region is not writable. In such a case the debug adapter should stop after hitting the first byte of memory that cannot be written and return the number of bytes written in the response via the 'offset' and 'bytesWritten' properties.\nIf false or missing, a debug adapter should attempt to verify the region is writable before writing, and fail the response if it is not."}, 'data': {'type': 'string', 'description': 'Bytes to write, encoded using base64.'}}
    __refs__ = set()
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, memoryReference, data, offset=None, allowPartial=None, update_ids_from_dap=False, **kwargs):
        """
        :param string memoryReference: Memory reference to the base location to which data should be written.
        :param string data: Bytes to write, encoded using base64.
        :param integer offset: Optional offset (in bytes) to be applied to the reference location before writing data. Can be negative.
        :param boolean allowPartial: Optional property to control partial writes. If true, the debug adapter should attempt to write memory even if the entire memory region is not writable. In such a case the debug adapter should stop after hitting the first byte of memory that cannot be written and return the number of bytes written in the response via the 'offset' and 'bytesWritten' properties.
        If false or missing, a debug adapter should attempt to verify the region is writable before writing, and fail the response if it is not.
        """
        self.memoryReference = memoryReference
        self.data = data
        self.offset = offset
        self.allowPartial = allowPartial
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        memoryReference = self.memoryReference
        data = self.data
        offset = self.offset
        allowPartial = self.allowPartial
        dct = {'memoryReference': memoryReference, 'data': data}
        if offset is not None:
            dct['offset'] = offset
        if allowPartial is not None:
            dct['allowPartial'] = allowPartial
        dct.update(self.kwargs)
        return dct