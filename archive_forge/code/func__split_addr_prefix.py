import functools
@classmethod
def _split_addr_prefix(cls, address):
    """Helper function to parse address of Network/Interface.

        Arg:
            address: Argument of Network/Interface.

        Returns:
            (addr, prefix) tuple.
        """
    if isinstance(address, (bytes, int)):
        return (address, cls._max_prefixlen)
    if not isinstance(address, tuple):
        address = _split_optional_netmask(address)
    if len(address) > 1:
        return address
    return (address[0], cls._max_prefixlen)