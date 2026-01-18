import functools
@classmethod
def _prefix_from_ip_string(cls, ip_str):
    """Turn a netmask/hostmask string into a prefix length

        Args:
            ip_str: The netmask/hostmask to be converted

        Returns:
            An integer, the prefix length.

        Raises:
            NetmaskValueError: If the input is not a valid netmask/hostmask
        """
    try:
        ip_int = cls._ip_int_from_string(ip_str)
    except AddressValueError:
        cls._report_invalid_netmask(ip_str)
    try:
        return cls._prefix_from_ip_int(ip_int)
    except ValueError:
        pass
    ip_int ^= cls._ALL_ONES
    try:
        return cls._prefix_from_ip_int(ip_int)
    except ValueError:
        cls._report_invalid_netmask(ip_str)