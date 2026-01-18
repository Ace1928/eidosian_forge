import functools
@staticmethod
def _split_scope_id(ip_str):
    """Helper function to parse IPv6 string address with scope id.

        See RFC 4007 for details.

        Args:
            ip_str: A string, the IPv6 address.

        Returns:
            (addr, scope_id) tuple.

        """
    addr, sep, scope_id = ip_str.partition('%')
    if not sep:
        scope_id = None
    elif not scope_id or '%' in scope_id:
        raise AddressValueError('Invalid IPv6 address: "%r"' % ip_str)
    return (addr, scope_id)