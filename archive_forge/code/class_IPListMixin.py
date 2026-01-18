import sys as _sys
from netaddr.core import (
from netaddr.strategy import ipv4 as _ipv4, ipv6 as _ipv6
class IPListMixin(object):
    """
    A mixin class providing shared list-like functionality to classes
    representing groups of IP addresses.

    """
    __slots__ = ()

    def __iter__(self):
        """
        :return: An iterator providing access to all `IPAddress` objects
            within range represented by this ranged IP object.
        """
        start_ip = IPAddress(self.first, self._module.version)
        end_ip = IPAddress(self.last, self._module.version)
        return iter_iprange(start_ip, end_ip)

    @property
    def size(self):
        """
        The total number of IP addresses within this ranged IP object.
        """
        return int(self.last - self.first + 1)

    def __len__(self):
        """
        :return: the number of IP addresses in this ranged IP object. Raises
            an `IndexError` if size > system max int (a Python 2.x
            limitation). Use the .size property for subnets of any size.
        """
        size = self.size
        if size > _sys.maxsize:
            raise IndexError('range contains more than %d (sys.maxsize) IP addresses! Use the .size property instead.' % _sys.maxsize)
        return size

    def __getitem__(self, index):
        """
        :return: The IP address(es) in this `IPNetwork` object referenced by
            index or slice. As slicing can produce large sequences of objects
            an iterator is returned instead of the more usual `list`.
        """
        item = None
        if hasattr(index, 'indices'):
            if self._module.version == 6:
                raise TypeError('IPv6 slices are not supported!')
            start, stop, step = index.indices(self.size)
            if start + step < 0 or step > stop:
                item = iter([IPAddress(self.first, self._module.version)])
            else:
                start_ip = IPAddress(self.first + start, self._module.version)
                one_before_stop = stop + (1 if step < 0 else -1)
                end_ip = IPAddress(self.first + one_before_stop, self._module.version)
                item = iter_iprange(start_ip, end_ip, step)
        else:
            try:
                index = int(index)
                if -self.size <= index < 0:
                    item = IPAddress(self.last + index + 1, self._module.version)
                elif 0 <= index <= self.size - 1:
                    item = IPAddress(self.first + index, self._module.version)
                else:
                    raise IndexError('index out range for address range size!')
            except ValueError:
                raise TypeError('unsupported index type %r!' % index)
        return item

    def __contains__(self, other):
        """
        :param other: an `IPAddress` or ranged IP object.

        :return: ``True`` if other falls within the boundary of this one,
            ``False`` otherwise.
        """
        if isinstance(other, BaseIP):
            if self._module.version != other._module.version:
                return False
            if isinstance(other, IPAddress):
                return other._value >= self.first and other._value <= self.last
            return other.first >= self.first and other.last <= self.last
        return IPAddress(other) in self

    def __bool__(self):
        """
        Ranged IP objects always represent a sequence of at least one IP
        address and are therefore always True in the boolean context.
        """
        return True