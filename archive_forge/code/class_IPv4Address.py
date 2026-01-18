import functools
class IPv4Address(_BaseV4, _BaseAddress):
    """Represent and manipulate single IPv4 Addresses."""
    __slots__ = ('_ip', '__weakref__')

    def __init__(self, address):
        """
        Args:
            address: A string or integer representing the IP

              Additionally, an integer can be passed, so
              IPv4Address('192.0.2.1') == IPv4Address(3221225985).
              or, more generally
              IPv4Address(int(IPv4Address('192.0.2.1'))) ==
                IPv4Address('192.0.2.1')

        Raises:
            AddressValueError: If ipaddress isn't a valid IPv4 address.

        """
        if isinstance(address, int):
            self._check_int_address(address)
            self._ip = address
            return
        if isinstance(address, bytes):
            self._check_packed_address(address, 4)
            self._ip = int.from_bytes(address)
            return
        addr_str = str(address)
        if '/' in addr_str:
            raise AddressValueError(f"Unexpected '/' in {address!r}")
        self._ip = self._ip_int_from_string(addr_str)

    @property
    def packed(self):
        """The binary representation of this address."""
        return v4_int_to_packed(self._ip)

    @property
    def is_reserved(self):
        """Test if the address is otherwise IETF reserved.

         Returns:
             A boolean, True if the address is within the
             reserved IPv4 Network range.

        """
        return self in self._constants._reserved_network

    @property
    @functools.lru_cache()
    def is_private(self):
        """Test if this address is allocated for private networks.

        Returns:
            A boolean, True if the address is reserved per
            iana-ipv4-special-registry.

        """
        return any((self in net for net in self._constants._private_networks))

    @property
    @functools.lru_cache()
    def is_global(self):
        return self not in self._constants._public_network and (not self.is_private)

    @property
    def is_multicast(self):
        """Test if the address is reserved for multicast use.

        Returns:
            A boolean, True if the address is multicast.
            See RFC 3171 for details.

        """
        return self in self._constants._multicast_network

    @property
    def is_unspecified(self):
        """Test if the address is unspecified.

        Returns:
            A boolean, True if this is the unspecified address as defined in
            RFC 5735 3.

        """
        return self == self._constants._unspecified_address

    @property
    def is_loopback(self):
        """Test if the address is a loopback address.

        Returns:
            A boolean, True if the address is a loopback per RFC 3330.

        """
        return self in self._constants._loopback_network

    @property
    def is_link_local(self):
        """Test if the address is reserved for link-local.

        Returns:
            A boolean, True if the address is link-local per RFC 3927.

        """
        return self in self._constants._linklocal_network