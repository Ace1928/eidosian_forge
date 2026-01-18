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
@dns.immutable.immutable
class ImmutableNode(Node):

    def __init__(self, node):
        super().__init__()
        self.rdatasets = tuple([dns.rdataset.ImmutableRdataset(rds) for rds in node.rdatasets])

    def find_rdataset(self, rdclass: dns.rdataclass.RdataClass, rdtype: dns.rdatatype.RdataType, covers: dns.rdatatype.RdataType=dns.rdatatype.NONE, create: bool=False) -> dns.rdataset.Rdataset:
        if create:
            raise TypeError('immutable')
        return super().find_rdataset(rdclass, rdtype, covers, False)

    def get_rdataset(self, rdclass: dns.rdataclass.RdataClass, rdtype: dns.rdatatype.RdataType, covers: dns.rdatatype.RdataType=dns.rdatatype.NONE, create: bool=False) -> Optional[dns.rdataset.Rdataset]:
        if create:
            raise TypeError('immutable')
        return super().get_rdataset(rdclass, rdtype, covers, False)

    def delete_rdataset(self, rdclass: dns.rdataclass.RdataClass, rdtype: dns.rdatatype.RdataType, covers: dns.rdatatype.RdataType=dns.rdatatype.NONE) -> None:
        raise TypeError('immutable')

    def replace_rdataset(self, replacement: dns.rdataset.Rdataset) -> None:
        raise TypeError('immutable')

    def is_immutable(self) -> bool:
        return True