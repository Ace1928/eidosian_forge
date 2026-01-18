import functools
class IPv6Address(_BaseV6, _BaseAddress):
    """Represent and manipulate single IPv6 Addresses."""
    __slots__ = ('_ip', '_scope_id', '__weakref__')

    def __init__(self, address):
        """Instantiate a new IPv6 address object.

        Args:
            address: A string or integer representing the IP

              Additionally, an integer can be passed, so
              IPv6Address('2001:db8::') ==
                IPv6Address(42540766411282592856903984951653826560)
              or, more generally
              IPv6Address(int(IPv6Address('2001:db8::'))) ==
                IPv6Address('2001:db8::')

        Raises:
            AddressValueError: If address isn't a valid IPv6 address.

        """
        if isinstance(address, int):
            self._check_int_address(address)
            self._ip = address
            self._scope_id = None
            return
        if isinstance(address, bytes):
            self._check_packed_address(address, 16)
            self._ip = int.from_bytes(address, 'big')
            self._scope_id = None
            return
        addr_str = str(address)
        if '/' in addr_str:
            raise AddressValueError(f"Unexpected '/' in {address!r}")
        addr_str, self._scope_id = self._split_scope_id(addr_str)
        self._ip = self._ip_int_from_string(addr_str)

    def __str__(self):
        ip_str = super().__str__()
        return ip_str + '%' + self._scope_id if self._scope_id else ip_str

    def __hash__(self):
        return hash((self._ip, self._scope_id))

    def __eq__(self, other):
        address_equal = super().__eq__(other)
        if address_equal is NotImplemented:
            return NotImplemented
        if not address_equal:
            return False
        return self._scope_id == getattr(other, '_scope_id', None)

    def __reduce__(self):
        return (self.__class__, (str(self),))

    @property
    def scope_id(self):
        """Identifier of a particular zone of the address's scope.

        See RFC 4007 for details.

        Returns:
            A string identifying the zone of the address if specified, else None.

        """
        return self._scope_id

    @property
    def packed(self):
        """The binary representation of this address."""
        return v6_int_to_packed(self._ip)

    @property
    def is_multicast(self):
        """Test if the address is reserved for multicast use.

        Returns:
            A boolean, True if the address is a multicast address.
            See RFC 2373 2.7 for details.

        """
        return self in self._constants._multicast_network

    @property
    def is_reserved(self):
        """Test if the address is otherwise IETF reserved.

        Returns:
            A boolean, True if the address is within one of the
            reserved IPv6 Network ranges.

        """
        return any((self in x for x in self._constants._reserved_networks))

    @property
    def is_link_local(self):
        """Test if the address is reserved for link-local.

        Returns:
            A boolean, True if the address is reserved per RFC 4291.

        """
        return self in self._constants._linklocal_network

    @property
    def is_site_local(self):
        """Test if the address is reserved for site-local.

        Note that the site-local address space has been deprecated by RFC 3879.
        Use is_private to test if this address is in the space of unique local
        addresses as defined by RFC 4193.

        Returns:
            A boolean, True if the address is reserved per RFC 3513 2.5.6.

        """
        return self in self._constants._sitelocal_network

    @property
    @functools.lru_cache()
    def is_private(self):
        """Test if this address is allocated for private networks.

        Returns:
            A boolean, True if the address is reserved per
            iana-ipv6-special-registry, or is ipv4_mapped and is
            reserved in the iana-ipv4-special-registry.

        """
        ipv4_mapped = self.ipv4_mapped
        if ipv4_mapped is not None:
            return ipv4_mapped.is_private
        return any((self in net for net in self._constants._private_networks))

    @property
    def is_global(self):
        """Test if this address is allocated for public networks.

        Returns:
            A boolean, true if the address is not reserved per
            iana-ipv6-special-registry.

        """
        return not self.is_private

    @property
    def is_unspecified(self):
        """Test if the address is unspecified.

        Returns:
            A boolean, True if this is the unspecified address as defined in
            RFC 2373 2.5.2.

        """
        return self._ip == 0

    @property
    def is_loopback(self):
        """Test if the address is a loopback address.

        Returns:
            A boolean, True if the address is a loopback address as defined in
            RFC 2373 2.5.3.

        """
        return self._ip == 1

    @property
    def ipv4_mapped(self):
        """Return the IPv4 mapped address.

        Returns:
            If the IPv6 address is a v4 mapped address, return the
            IPv4 mapped address. Return None otherwise.

        """
        if self._ip >> 32 != 65535:
            return None
        return IPv4Address(self._ip & 4294967295)

    @property
    def teredo(self):
        """Tuple of embedded teredo IPs.

        Returns:
            Tuple of the (server, client) IPs or None if the address
            doesn't appear to be a teredo address (doesn't start with
            2001::/32)

        """
        if self._ip >> 96 != 536936448:
            return None
        return (IPv4Address(self._ip >> 64 & 4294967295), IPv4Address(~self._ip & 4294967295))

    @property
    def sixtofour(self):
        """Return the IPv4 6to4 embedded address.

        Returns:
            The IPv4 6to4-embedded address if present or None if the
            address doesn't appear to contain a 6to4 embedded address.

        """
        if self._ip >> 112 != 8194:
            return None
        return IPv4Address(self._ip >> 80 & 4294967295)