import functools
class IPv6Network(_BaseV6, _BaseNetwork):
    """This class represents and manipulates 128-bit IPv6 networks.

    Attributes: [examples for IPv6('2001:db8::1000/124')]
        .network_address: IPv6Address('2001:db8::1000')
        .hostmask: IPv6Address('::f')
        .broadcast_address: IPv6Address('2001:db8::100f')
        .netmask: IPv6Address('ffff:ffff:ffff:ffff:ffff:ffff:ffff:fff0')
        .prefixlen: 124

    """
    _address_class = IPv6Address

    def __init__(self, address, strict=True):
        """Instantiate a new IPv6 Network object.

        Args:
            address: A string or integer representing the IPv6 network or the
              IP and prefix/netmask.
              '2001:db8::/128'
              '2001:db8:0000:0000:0000:0000:0000:0000/128'
              '2001:db8::'
              are all functionally the same in IPv6.  That is to say,
              failing to provide a subnetmask will create an object with
              a mask of /128.

              Additionally, an integer can be passed, so
              IPv6Network('2001:db8::') ==
                IPv6Network(42540766411282592856903984951653826560)
              or, more generally
              IPv6Network(int(IPv6Network('2001:db8::'))) ==
                IPv6Network('2001:db8::')

            strict: A boolean. If true, ensure that we have been passed
              A true network address, eg, 2001:db8::1000/124 and not an
              IP address on a network, eg, 2001:db8::1/124.

        Raises:
            AddressValueError: If address isn't a valid IPv6 address.
            NetmaskValueError: If the netmask isn't valid for
              an IPv6 address.
            ValueError: If strict was True and a network address was not
              supplied.
        """
        addr, mask = self._split_addr_prefix(address)
        self.network_address = IPv6Address(addr)
        self.netmask, self._prefixlen = self._make_netmask(mask)
        packed = int(self.network_address)
        if packed & int(self.netmask) != packed:
            if strict:
                raise ValueError('%s has host bits set' % self)
            else:
                self.network_address = IPv6Address(packed & int(self.netmask))
        if self._prefixlen == self._max_prefixlen - 1:
            self.hosts = self.__iter__
        elif self._prefixlen == self._max_prefixlen:
            self.hosts = lambda: [IPv6Address(addr)]

    def hosts(self):
        """Generate Iterator over usable hosts in a network.

          This is like __iter__ except it doesn't return the
          Subnet-Router anycast address.

        """
        network = int(self.network_address)
        broadcast = int(self.broadcast_address)
        for x in range(network + 1, broadcast + 1):
            yield self._address_class(x)

    @property
    def is_site_local(self):
        """Test if the address is reserved for site-local.

        Note that the site-local address space has been deprecated by RFC 3879.
        Use is_private to test if this address is in the space of unique local
        addresses as defined by RFC 4193.

        Returns:
            A boolean, True if the address is reserved per RFC 3513 2.5.6.

        """
        return self.network_address.is_site_local and self.broadcast_address.is_site_local