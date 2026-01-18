import functools
def _check_packed_address(self, address, expected_len):
    address_len = len(address)
    if address_len != expected_len:
        msg = '%r (len %d != %d) is not permitted as an IPv%d address'
        raise AddressValueError(msg % (address, address_len, expected_len, self._version))