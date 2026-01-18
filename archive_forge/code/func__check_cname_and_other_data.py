import re
import sys
from typing import Any, Iterable, List, Optional, Set, Tuple, Union
import dns.exception
import dns.grange
import dns.name
import dns.node
import dns.rdata
import dns.rdataclass
import dns.rdatatype
import dns.rdtypes.ANY.SOA
import dns.rrset
import dns.tokenizer
import dns.transaction
import dns.ttl
def _check_cname_and_other_data(txn, name, rdataset):
    rdataset_kind = dns.node.NodeKind.classify_rdataset(rdataset)
    node = txn.get_node(name)
    if node is None:
        return
    node_kind = node.classify()
    if node_kind == dns.node.NodeKind.CNAME and rdataset_kind == dns.node.NodeKind.REGULAR:
        raise CNAMEAndOtherData('rdataset type is not compatible with a CNAME node')
    elif node_kind == dns.node.NodeKind.REGULAR and rdataset_kind == dns.node.NodeKind.CNAME:
        raise CNAMEAndOtherData('CNAME rdataset is not compatible with a regular data node')