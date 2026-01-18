import contextlib
import io
import os
import struct
from typing import (
import dns.exception
import dns.grange
import dns.immutable
import dns.name
import dns.node
import dns.rdata
import dns.rdataclass
import dns.rdataset
import dns.rdatatype
import dns.rdtypes.ANY.SOA
import dns.rdtypes.ANY.ZONEMD
import dns.rrset
import dns.tokenizer
import dns.transaction
import dns.ttl
import dns.zonefile
from dns.zonetypes import DigestHashAlgorithm, DigestScheme, _digest_hashers
def _maybe_cow(self, name: dns.name.Name) -> dns.node.Node:
    name = self._validate_name(name)
    node = self.nodes.get(name)
    if node is None or name not in self.changed:
        new_node = self.zone.node_factory()
        if hasattr(new_node, 'id'):
            new_node.id = self.id
        if node is not None:
            new_node.rdatasets.extend(node.rdatasets)
        self.nodes[name] = new_node
        self.changed.add(name)
        return new_node
    else:
        return node