import functools
class _IPAddressBase:
    """The mother class."""
    __slots__ = ()

    @property
    def exploded(self):
        """Return the longhand version of the IP address as a string."""
        return self._explode_shorthand_ip_string()

    @property
    def compressed(self):
        """Return the shorthand version of the IP address as a string."""
        return str(self)

    @property
    def reverse_pointer(self):
        """The name of the reverse DNS pointer for the IP address, e.g.:
            >>> ipaddress.ip_address("127.0.0.1").reverse_pointer
            '1.0.0.127.in-addr.arpa'
            >>> ipaddress.ip_address("2001:db8::1").reverse_pointer
            '1.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.8.b.d.0.1.0.0.2.ip6.arpa'

        """
        return self._reverse_pointer()

    @property
    def version(self):
        msg = '%200s has no version specified' % (type(self),)
        raise NotImplementedError(msg)

    def _check_int_address(self, address):
        if address < 0:
            msg = '%d (< 0) is not permitted as an IPv%d address'
            raise AddressValueError(msg % (address, self._version))
        if address > self._ALL_ONES:
            msg = '%d (>= 2**%d) is not permitted as an IPv%d address'
            raise AddressValueError(msg % (address, self._max_prefixlen, self._version))

    def _check_packed_address(self, address, expected_len):
        address_len = len(address)
        if address_len != expected_len:
            msg = '%r (len %d != %d) is not permitted as an IPv%d address'
            raise AddressValueError(msg % (address, address_len, expected_len, self._version))

    @classmethod
    def _ip_int_from_prefix(cls, prefixlen):
        """Turn the prefix length into a bitwise netmask

        Args:
            prefixlen: An integer, the prefix length.

        Returns:
            An integer.

        """
        return cls._ALL_ONES ^ cls._ALL_ONES >> prefixlen

    @classmethod
    def _prefix_from_ip_int(cls, ip_int):
        """Return prefix length from the bitwise netmask.

        Args:
            ip_int: An integer, the netmask in expanded bitwise format

        Returns:
            An integer, the prefix length.

        Raises:
            ValueError: If the input intermingles zeroes & ones
        """
        trailing_zeroes = _count_righthand_zero_bits(ip_int, cls._max_prefixlen)
        prefixlen = cls._max_prefixlen - trailing_zeroes
        leading_ones = ip_int >> trailing_zeroes
        all_ones = (1 << prefixlen) - 1
        if leading_ones != all_ones:
            byteslen = cls._max_prefixlen // 8
            details = ip_int.to_bytes(byteslen, 'big')
            msg = 'Netmask pattern %r mixes zeroes & ones'
            raise ValueError(msg % details)
        return prefixlen

    @classmethod
    def _report_invalid_netmask(cls, netmask_str):
        msg = '%r is not a valid netmask' % netmask_str
        raise NetmaskValueError(msg) from None

    @classmethod
    def _prefix_from_prefix_string(cls, prefixlen_str):
        """Return prefix length from a numeric string

        Args:
            prefixlen_str: The string to be converted

        Returns:
            An integer, the prefix length.

        Raises:
            NetmaskValueError: If the input is not a valid netmask
        """
        if not (prefixlen_str.isascii() and prefixlen_str.isdigit()):
            cls._report_invalid_netmask(prefixlen_str)
        try:
            prefixlen = int(prefixlen_str)
        except ValueError:
            cls._report_invalid_netmask(prefixlen_str)
        if not 0 <= prefixlen <= cls._max_prefixlen:
            cls._report_invalid_netmask(prefixlen_str)
        return prefixlen

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

    def __reduce__(self):
        return (self.__class__, (str(self),))