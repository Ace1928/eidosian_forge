import functools
class IPv4Interface(IPv4Address):

    def __init__(self, address):
        addr, mask = self._split_addr_prefix(address)
        IPv4Address.__init__(self, addr)
        self.network = IPv4Network((addr, mask), strict=False)
        self.netmask = self.network.netmask
        self._prefixlen = self.network._prefixlen

    @functools.cached_property
    def hostmask(self):
        return self.network.hostmask

    def __str__(self):
        return '%s/%d' % (self._string_from_ip_int(self._ip), self._prefixlen)

    def __eq__(self, other):
        address_equal = IPv4Address.__eq__(self, other)
        if address_equal is NotImplemented or not address_equal:
            return address_equal
        try:
            return self.network == other.network
        except AttributeError:
            return False

    def __lt__(self, other):
        address_less = IPv4Address.__lt__(self, other)
        if address_less is NotImplemented:
            return NotImplemented
        try:
            return self.network < other.network or (self.network == other.network and address_less)
        except AttributeError:
            return False

    def __hash__(self):
        return hash((self._ip, self._prefixlen, int(self.network.network_address)))
    __reduce__ = _IPAddressBase.__reduce__

    @property
    def ip(self):
        return IPv4Address(self._ip)

    @property
    def with_prefixlen(self):
        return '%s/%s' % (self._string_from_ip_int(self._ip), self._prefixlen)

    @property
    def with_netmask(self):
        return '%s/%s' % (self._string_from_ip_int(self._ip), self.netmask)

    @property
    def with_hostmask(self):
        return '%s/%s' % (self._string_from_ip_int(self._ip), self.hostmask)