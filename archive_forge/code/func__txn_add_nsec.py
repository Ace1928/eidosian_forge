import base64
import contextlib
import functools
import hashlib
import struct
import time
from datetime import datetime
from typing import Callable, Dict, List, Optional, Set, Tuple, Union, cast
import dns._features
import dns.exception
import dns.name
import dns.node
import dns.rdata
import dns.rdataclass
import dns.rdataset
import dns.rdatatype
import dns.rrset
import dns.transaction
import dns.zone
from dns.dnssectypes import Algorithm, DSDigest, NSEC3Hash
from dns.exception import (  # pylint: disable=W0611
from dns.rdtypes.ANY.CDNSKEY import CDNSKEY
from dns.rdtypes.ANY.CDS import CDS
from dns.rdtypes.ANY.DNSKEY import DNSKEY
from dns.rdtypes.ANY.DS import DS
from dns.rdtypes.ANY.NSEC import NSEC, Bitmap
from dns.rdtypes.ANY.NSEC3PARAM import NSEC3PARAM
from dns.rdtypes.ANY.RRSIG import RRSIG, sigtime_to_posixtime
from dns.rdtypes.dnskeybase import Flag
def _txn_add_nsec(txn: dns.transaction.Transaction, name: dns.name.Name, next_secure: Optional[dns.name.Name], rdclass: dns.rdataclass.RdataClass, ttl: int, rrset_signer: Optional[RRsetSigner]=None) -> None:
    """NSEC zone signer helper"""
    mandatory_types = set([dns.rdatatype.RdataType.RRSIG, dns.rdatatype.RdataType.NSEC])
    node = txn.get_node(name)
    if node and next_secure:
        types = set([rdataset.rdtype for rdataset in node.rdatasets]) | mandatory_types
        windows = Bitmap.from_rdtypes(list(types))
        rrset = dns.rrset.from_rdata(name, ttl, NSEC(rdclass=rdclass, rdtype=dns.rdatatype.RdataType.NSEC, next=next_secure, windows=windows))
        txn.add(rrset)
        if rrset_signer:
            rrset_signer(txn, rrset)