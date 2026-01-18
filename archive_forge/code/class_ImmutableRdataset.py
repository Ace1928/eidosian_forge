import io
import random
import struct
from typing import Any, Collection, Dict, List, Optional, Union, cast
import dns.exception
import dns.immutable
import dns.name
import dns.rdata
import dns.rdataclass
import dns.rdatatype
import dns.renderer
import dns.set
import dns.ttl
@dns.immutable.immutable
class ImmutableRdataset(Rdataset):
    """An immutable DNS rdataset."""
    _clone_class = Rdataset

    def __init__(self, rdataset: Rdataset):
        """Create an immutable rdataset from the specified rdataset."""
        super().__init__(rdataset.rdclass, rdataset.rdtype, rdataset.covers, rdataset.ttl)
        self.items = dns.immutable.Dict(rdataset.items)

    def update_ttl(self, ttl):
        raise TypeError('immutable')

    def add(self, rd, ttl=None):
        raise TypeError('immutable')

    def union_update(self, other):
        raise TypeError('immutable')

    def intersection_update(self, other):
        raise TypeError('immutable')

    def update(self, other):
        raise TypeError('immutable')

    def __delitem__(self, i):
        raise TypeError('immutable')

    def __ior__(self, other):
        raise TypeError('immutable')

    def __iand__(self, other):
        raise TypeError('immutable')

    def __iadd__(self, other):
        raise TypeError('immutable')

    def __isub__(self, other):
        raise TypeError('immutable')

    def clear(self):
        raise TypeError('immutable')

    def __copy__(self):
        return ImmutableRdataset(super().copy())

    def copy(self):
        return ImmutableRdataset(super().copy())

    def union(self, other):
        return ImmutableRdataset(super().union(other))

    def intersection(self, other):
        return ImmutableRdataset(super().intersection(other))

    def difference(self, other):
        return ImmutableRdataset(super().difference(other))

    def symmetric_difference(self, other):
        return ImmutableRdataset(super().symmetric_difference(other))