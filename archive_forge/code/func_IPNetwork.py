import struct
import sys
def IPNetwork(address, version=None, strict=False):
    """Take an IP string/int and return an object of the correct type.

    Args:
        address: A string or integer, the IP address.  Either IPv4 or
          IPv6 addresses may be supplied; integers less than 2**32 will
          be considered to be IPv4 by default.
        version: An Integer, if set, don't try to automatically
          determine what the IP address type is. important for things
          like IPNetwork(1), which could be IPv4, '0.0.0.1/32', or IPv6,
          '::1/128'.

    Returns:
        An IPv4Network or IPv6Network object.

    Raises:
        ValueError: if the string passed isn't either a v4 or a v6
          address. Or if a strict network was requested and a strict
          network wasn't given.

    """
    if version:
        if version == 4:
            return IPv4Network(address, strict)
        elif version == 6:
            return IPv6Network(address, strict)
    try:
        return IPv4Network(address, strict)
    except (AddressValueError, NetmaskValueError):
        pass
    try:
        return IPv6Network(address, strict)
    except (AddressValueError, NetmaskValueError):
        pass
    raise ValueError('%r does not appear to be an IPv4 or IPv6 network' % address)