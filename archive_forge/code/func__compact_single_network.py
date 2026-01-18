import itertools as _itertools
import sys as _sys
from netaddr.ip import IPNetwork, IPAddress, IPRange, cidr_merge, cidr_exclude, iprange_to_cidrs
def _compact_single_network(self, added_network):
    """
        Same as compact(), but assume that added_network is the only change and
        that this IPSet was properly compacted before added_network was added.
        This allows to perform compaction much faster. added_network must
        already be present in self._cidrs.
        """
    added_first = added_network.first
    added_last = added_network.last
    added_version = added_network.version
    if added_network._prefixlen == added_network._module.width:
        for potential_supernet in added_network.supernet():
            if potential_supernet in self._cidrs:
                del self._cidrs[added_network]
                return
    else:
        to_remove = []
        for cidr in self._cidrs:
            if cidr._module.version != added_version or cidr == added_network:
                continue
            first = cidr.first
            last = cidr.last
            if first >= added_first and last <= added_last:
                to_remove.append(cidr)
            elif first <= added_first and last >= added_last:
                del self._cidrs[added_network]
                assert not to_remove
                return
        for item in to_remove:
            del self._cidrs[item]
    shift_width = added_network._module.width - added_network.prefixlen
    while added_network.prefixlen != 0:
        the_bit = added_network._value >> shift_width & 1
        if the_bit:
            candidate = added_network.previous()
        else:
            candidate = added_network.next()
        if candidate not in self._cidrs:
            return
        del self._cidrs[candidate]
        del self._cidrs[added_network]
        added_network.prefixlen -= 1
        shift_width += 1
        added_network._value = added_network._value >> shift_width << shift_width
        self._cidrs[added_network] = True