from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class ReadMemoryResponseBody(BaseSchema):
    """
    "body" of ReadMemoryResponse

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'address': {'type': 'string', 'description': "The address of the first byte of data returned.\nTreated as a hex value if prefixed with '0x', or as a decimal value otherwise."}, 'unreadableBytes': {'type': 'integer', 'description': "The number of unreadable bytes encountered after the last successfully read byte.\nThis can be used to determine the number of bytes that must be skipped before a subsequent 'readMemory' request will succeed."}, 'data': {'type': 'string', 'description': 'The bytes read from memory, encoded using base64.'}}
    __refs__ = set()
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, address, unreadableBytes=None, data=None, update_ids_from_dap=False, **kwargs):
        """
        :param string address: The address of the first byte of data returned.
        Treated as a hex value if prefixed with '0x', or as a decimal value otherwise.
        :param integer unreadableBytes: The number of unreadable bytes encountered after the last successfully read byte.
        This can be used to determine the number of bytes that must be skipped before a subsequent 'readMemory' request will succeed.
        :param string data: The bytes read from memory, encoded using base64.
        """
        self.address = address
        self.unreadableBytes = unreadableBytes
        self.data = data
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        address = self.address
        unreadableBytes = self.unreadableBytes
        data = self.data
        dct = {'address': address}
        if unreadableBytes is not None:
            dct['unreadableBytes'] = unreadableBytes
        if data is not None:
            dct['data'] = data
        dct.update(self.kwargs)
        return dct