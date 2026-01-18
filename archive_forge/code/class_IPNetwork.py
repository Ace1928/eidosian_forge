import sys as _sys
from netaddr.core import (
from netaddr.strategy import ipv4 as _ipv4, ipv6 as _ipv6
class IPNetwork(BaseIP, IPListMixin):
    """
    An IPv4 or IPv6 network or subnet.

    A combination of an IP address and a network mask.

    Accepts CIDR and several related variants :

    a) Standard CIDR::

        x.x.x.x/y -> 192.0.2.0/24
        x::/y -> fe80::/10

    b) Hybrid CIDR format (netmask address instead of prefix), where 'y'        address represent a valid netmask::

        x.x.x.x/y.y.y.y -> 192.0.2.0/255.255.255.0
        x::/y:: -> fe80::/ffc0::

    c) ACL hybrid CIDR format (hostmask address instead of prefix like        Cisco's ACL bitmasks), where 'y' address represent a valid netmask::

        x.x.x.x/y.y.y.y -> 192.0.2.0/0.0.0.255
        x::/y:: -> fe80::/3f:ffff:ffff:ffff:ffff:ffff:ffff:ffff
    
    .. versionchanged:: 1.0.0
        Removed the ``implicit_prefix`` switch that used to enable the abbreviated CIDR
        format support, use :func:`cidr_abbrev_to_verbose` if you need this behavior.

    .. versionchanged:: 1.1.0
        Removed partial IPv4 address support accidentally left when making 1.0.0 release.
        Use :func:`expand_partial_ipv4_address` if you need this behavior.
    """
    __slots__ = ('_prefixlen',)

    def __init__(self, addr, version=None, flags=0):
        """
        Constructor.

        :param addr: an IPv4 or IPv6 address with optional CIDR prefix,
            netmask or hostmask. May be an IP address in presentation
            (string) format, an tuple containing and integer address and a
            network prefix, or another IPAddress/IPNetwork object (copy
            construction).

        :param version: (optional) optimizes version detection if specified
            and distinguishes between IPv4 and IPv6 for addresses with an
            equivalent integer value.

        :param flags: (optional) decides which rules are applied to the
            interpretation of the addr value. Currently only supports the
            :data:`NOHOST` option.

            >>> IPNetwork('1.2.3.4/24')
            IPNetwork('1.2.3.4/24')
            >>> IPNetwork('1.2.3.4/24', flags=NOHOST)
            IPNetwork('1.2.3.0/24')
        """
        super(IPNetwork, self).__init__()
        if flags & ~NOHOST:
            raise ValueError('Unrecognized IPAddress flags value: %s' % (flags,))
        value, prefixlen, module = (None, None, None)
        if hasattr(addr, '_prefixlen'):
            value = addr._value
            module = addr._module
            prefixlen = addr._prefixlen
        elif hasattr(addr, '_value'):
            value = addr._value
            module = addr._module
            prefixlen = module.width
        elif version == 4:
            value, prefixlen = parse_ip_network(_ipv4, addr, flags=flags)
            module = _ipv4
        elif version == 6:
            value, prefixlen = parse_ip_network(_ipv6, addr, flags=flags)
            module = _ipv6
        else:
            if version is not None:
                raise ValueError('%r is an invalid IP version!' % version)
            try:
                module = _ipv4
                value, prefixlen = parse_ip_network(module, addr, flags)
            except AddrFormatError:
                try:
                    module = _ipv6
                    value, prefixlen = parse_ip_network(module, addr, flags)
                except AddrFormatError:
                    pass
                if value is None:
                    raise AddrFormatError('invalid IPNetwork %s' % (addr,))
        self._value = value
        self._prefixlen = prefixlen
        self._module = module

    def __getstate__(self):
        """:return: Pickled state of an `IPNetwork` object."""
        return (self._value, self._prefixlen, self._module.version)

    def __setstate__(self, state):
        """
        :param state: data used to unpickle a pickled `IPNetwork` object.

        """
        value, prefixlen, version = state
        self._value = value
        if version == 4:
            self._module = _ipv4
        elif version == 6:
            self._module = _ipv6
        else:
            raise ValueError('unpickling failed for object state %s' % (state,))
        if 0 <= prefixlen <= self._module.width:
            self._prefixlen = prefixlen
        else:
            raise ValueError('unpickling failed for object state %s' % (state,))

    def _set_prefixlen(self, value):
        if not isinstance(value, int):
            raise TypeError('int argument expected, not %s' % type(value))
        if not 0 <= value <= self._module.width:
            raise AddrFormatError('invalid prefix for an %s address!' % self._module.family_name)
        self._prefixlen = value
    prefixlen = property(lambda self: self._prefixlen, _set_prefixlen, doc='size of the bitmask used to separate the network from the host bits')

    @property
    def ip(self):
        """
        The IP address of this `IPNetwork` object. This is may or may not be
        the same as the network IP address which varies according to the value
        of the CIDR subnet prefix.
        """
        return IPAddress(self._value, self._module.version)

    @property
    def network(self):
        """The network address of this `IPNetwork` object."""
        return IPAddress(self._value & self._netmask_int, self._module.version)

    @property
    def broadcast(self):
        """The broadcast address of this `IPNetwork` object."""
        if self._module.width - self._prefixlen <= 1:
            return None
        else:
            return IPAddress(self._value | self._hostmask_int, self._module.version)

    @property
    def first(self):
        """
        The integer value of first IP address found within this `IPNetwork`
        object.
        """
        return self._value & (self._module.max_int ^ self._hostmask_int)

    @property
    def last(self):
        """
        The integer value of last IP address found within this `IPNetwork`
        object.
        """
        hostmask = (1 << self._module.width - self._prefixlen) - 1
        return self._value | hostmask

    @property
    def netmask(self):
        """The subnet mask of this `IPNetwork` object."""
        netmask = self._module.max_int ^ self._hostmask_int
        return IPAddress(netmask, self._module.version)

    @netmask.setter
    def netmask(self, value):
        """Set the prefixlen using a subnet mask"""
        ip = IPAddress(value)
        if ip.version != self.version:
            raise ValueError('IP version mismatch: %s and %s' % (ip, self))
        if not ip.is_netmask():
            raise ValueError('Invalid subnet mask specified: %s' % str(value))
        self.prefixlen = ip.netmask_bits()

    @property
    def _netmask_int(self):
        """Same as self.netmask, but in integer format"""
        return self._module.max_int ^ self._hostmask_int

    @property
    def hostmask(self):
        """The host mask of this `IPNetwork` object."""
        hostmask = (1 << self._module.width - self._prefixlen) - 1
        return IPAddress(hostmask, self._module.version)

    @property
    def _hostmask_int(self):
        """Same as self.hostmask, but in integer format"""
        return (1 << self._module.width - self._prefixlen) - 1

    @property
    def cidr(self):
        """
        The true CIDR address for this `IPNetwork` object which omits any
        host bits to the right of the CIDR subnet prefix.
        """
        return IPNetwork((self._value & self._netmask_int, self._prefixlen), version=self._module.version)

    def __iadd__(self, num):
        """
        Increases the value of this `IPNetwork` object by the current size
        multiplied by ``num``.

        An `IndexError` is raised if result exceeds maximum IP address value
        or is less than zero.

        :param num: (optional) number of `IPNetwork` blocks to increment             this IPNetwork's value by.
        """
        new_value = int(self.network) + self.size * num
        if new_value + (self.size - 1) > self._module.max_int:
            raise IndexError('increment exceeds address boundary!')
        if new_value < 0:
            raise IndexError('increment is less than zero!')
        self._value = new_value
        return self

    def __isub__(self, num):
        """
        Decreases the value of this `IPNetwork` object by the current size
        multiplied by ``num``.

        An `IndexError` is raised if result is less than zero or exceeds
        maximum IP address value.

        :param num: (optional) number of `IPNetwork` blocks to decrement             this IPNetwork's value by.
        """
        new_value = int(self.network) - self.size * num
        if new_value < 0:
            raise IndexError('decrement is less than zero!')
        if new_value + (self.size - 1) > self._module.max_int:
            raise IndexError('decrement exceeds address boundary!')
        self._value = new_value
        return self

    def __contains__(self, other):
        """
        :param other: an `IPAddress` or ranged IP object.

        :return: ``True`` if other falls within the boundary of this one,
            ``False`` otherwise.
        """
        if isinstance(other, BaseIP):
            if self._module.version != other._module.version:
                return False
            shiftwidth = self._module.width - self._prefixlen
            self_net = self._value >> shiftwidth
            if isinstance(other, IPRange):
                return self_net << shiftwidth <= other._start._value and self_net + 1 << shiftwidth > other._end._value
            other_net = other._value >> shiftwidth
            if isinstance(other, IPAddress):
                return other_net == self_net
            if isinstance(other, IPNetwork):
                return self_net == other_net and self._prefixlen <= other._prefixlen
        return IPNetwork(other) in self

    def key(self):
        """
        :return: A key tuple used to uniquely identify this `IPNetwork`.
        """
        return (self._module.version, self.first, self.last)

    def sort_key(self):
        """
        :return: A key tuple used to compare and sort this `IPNetwork` correctly.
        """
        net_size_bits = self._prefixlen - 1
        first = self._value & (self._module.max_int ^ self._hostmask_int)
        host_bits = self._value - first
        return (self._module.version, first, net_size_bits, host_bits)

    def ipv4(self):
        """
        :return: A numerically equivalent version 4 `IPNetwork` object.             Raises an `AddrConversionError` if IPv6 address cannot be             converted to IPv4.
        """
        ip = None
        klass = self.__class__
        if self._module.version == 4:
            ip = klass('%s/%d' % (self.ip, self.prefixlen))
        elif self._module.version == 6:
            if 0 <= self._value <= _ipv4.max_int:
                addr = _ipv4.int_to_str(self._value)
                ip = klass('%s/%d' % (addr, self.prefixlen - 96))
            elif _ipv4.max_int <= self._value <= 281474976710655:
                addr = _ipv4.int_to_str(self._value - 281470681743360)
                ip = klass('%s/%d' % (addr, self.prefixlen - 96))
            else:
                raise AddrConversionError('IPv6 address %s unsuitable for conversion to IPv4!' % self)
        return ip

    def ipv6(self, ipv4_compatible=False):
        """
        .. note:: the IPv4-mapped IPv6 address format is now considered         deprecated. See RFC 4291 or later for details.

        :param ipv4_compatible: If ``True`` returns an IPv4-mapped address
            (::ffff:x.x.x.x), an IPv4-compatible (::x.x.x.x) address
            otherwise. Default: False (IPv4-mapped).

        :return: A numerically equivalent version 6 `IPNetwork` object.
        """
        ip = None
        klass = self.__class__
        if self._module.version == 6:
            if ipv4_compatible and 281470681743360 <= self._value <= 281474976710655:
                ip = klass((self._value - 281470681743360, self._prefixlen), version=6)
            else:
                ip = klass((self._value, self._prefixlen), version=6)
        elif self._module.version == 4:
            if ipv4_compatible:
                ip = klass((self._value, self._prefixlen + 96), version=6)
            else:
                ip = klass((281470681743360 + self._value, self._prefixlen + 96), version=6)
        return ip

    def previous(self, step=1):
        """
        :param step: the number of IP subnets between this `IPNetwork` object
            and the expected subnet. Default: 1 (the previous IP subnet).

        :return: The adjacent subnet preceding this `IPNetwork` object.
        """
        ip_copy = self.__class__('%s/%d' % (self.network, self.prefixlen), self._module.version)
        ip_copy -= step
        return ip_copy

    def next(self, step=1):
        """
        :param step: the number of IP subnets between this `IPNetwork` object
            and the expected subnet. Default: 1 (the next IP subnet).

        :return: The adjacent subnet succeeding this `IPNetwork` object.
        """
        ip_copy = self.__class__('%s/%d' % (self.network, self.prefixlen), self._module.version)
        ip_copy += step
        return ip_copy

    def supernet(self, prefixlen=0):
        """
        Provides a list of supernets for this `IPNetwork` object between the
        size of the current prefix and (if specified) an endpoint prefix.

        :param prefixlen: (optional) a CIDR prefix for the maximum supernet.
            Default: 0 - returns all possible supernets.

        :return: a tuple of supernet `IPNetwork` objects.
        """
        if not 0 <= prefixlen <= self._module.width:
            raise ValueError('CIDR prefix /%d invalid for IPv%d!' % (prefixlen, self._module.version))
        supernets = []
        supernet = self.cidr
        supernet._prefixlen = prefixlen
        while supernet._prefixlen != self._prefixlen:
            supernets.append(supernet.cidr)
            supernet._prefixlen += 1
        return supernets

    def subnet(self, prefixlen, count=None, fmt=None):
        """
        A generator that divides up this IPNetwork's subnet into smaller
        subnets based on a specified CIDR prefix.

        :param prefixlen: a CIDR prefix indicating size of subnets to be
            returned.

        :param count: (optional) number of consecutive IP subnets to be
            returned.

        :return: an iterator containing IPNetwork subnet objects.
        """
        if not 0 <= self.prefixlen <= self._module.width:
            raise ValueError('CIDR prefix /%d invalid for IPv%d!' % (prefixlen, self._module.version))
        if not self.prefixlen <= prefixlen:
            return
        width = self._module.width
        max_subnets = 2 ** (width - self.prefixlen) // 2 ** (width - prefixlen)
        if count is None:
            count = max_subnets
        if not 1 <= count <= max_subnets:
            raise ValueError('count outside of current IP subnet boundary!')
        base_subnet = self._module.int_to_str(self.first)
        i = 0
        while i < count:
            subnet = self.__class__('%s/%d' % (base_subnet, prefixlen), self._module.version)
            subnet.value += subnet.size * i
            subnet.prefixlen = prefixlen
            i += 1
            yield subnet

    def iter_hosts(self):
        """
        A generator that provides all the IP addresses that can be assigned
        to hosts within the range of this IP object's subnet.

        - for IPv4, the network and broadcast addresses are excluded, excepted           when using /31 or /32 subnets as per RFC 3021.

        - for IPv6, only Subnet-Router anycast address (first address in the           network) is excluded as per RFC 4291 section 2.6.1, excepted when using           /127 or /128 subnets as per RFC 6164.

        :return: an IPAddress iterator
        """
        first_usable_address, last_usable_address = self._usable_range()
        return iter_iprange(IPAddress(first_usable_address, self._module.version), IPAddress(last_usable_address, self._module.version))

    def _usable_range(self):
        if self.size >= 4:
            first_usable_address = self.first + 1
            if self._module.version == 4:
                last_usable_address = self.last - 1
            else:
                last_usable_address = self.last
            return (first_usable_address, last_usable_address)
        else:
            return (self.first, self.last)

    def __str__(self):
        """:return: this IPNetwork in CIDR format"""
        addr = self._module.int_to_str(self._value)
        return '%s/%s' % (addr, self.prefixlen)

    def __repr__(self):
        """:return: Python statement to create an equivalent object"""
        return "%s('%s')" % (self.__class__.__name__, self)