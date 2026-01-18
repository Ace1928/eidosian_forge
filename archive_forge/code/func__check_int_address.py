import functools
def _check_int_address(self, address):
    if address < 0:
        msg = '%d (< 0) is not permitted as an IPv%d address'
        raise AddressValueError(msg % (address, self._version))
    if address > self._ALL_ONES:
        msg = '%d (>= 2**%d) is not permitted as an IPv%d address'
        raise AddressValueError(msg % (address, self._max_prefixlen, self._version))