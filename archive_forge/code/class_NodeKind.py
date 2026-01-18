import enum
import io
from typing import Any, Dict, Optional
import dns.immutable
import dns.name
import dns.rdataclass
import dns.rdataset
import dns.rdatatype
import dns.renderer
import dns.rrset
@enum.unique
class NodeKind(enum.Enum):
    """Rdatasets in nodes"""
    REGULAR = 0
    NEUTRAL = 1
    CNAME = 2

    @classmethod
    def classify(cls, rdtype: dns.rdatatype.RdataType, covers: dns.rdatatype.RdataType) -> 'NodeKind':
        if _matches_type_or_its_signature(_cname_types, rdtype, covers):
            return NodeKind.CNAME
        elif _matches_type_or_its_signature(_neutral_types, rdtype, covers):
            return NodeKind.NEUTRAL
        else:
            return NodeKind.REGULAR

    @classmethod
    def classify_rdataset(cls, rdataset: dns.rdataset.Rdataset) -> 'NodeKind':
        return cls.classify(rdataset.rdtype, rdataset.covers)