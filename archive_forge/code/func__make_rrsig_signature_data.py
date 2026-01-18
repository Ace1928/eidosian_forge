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
def _make_rrsig_signature_data(rrset: Union[dns.rrset.RRset, Tuple[dns.name.Name, dns.rdataset.Rdataset]], rrsig: RRSIG, origin: Optional[dns.name.Name]=None) -> bytes:
    """Create signature rdata.

    *rrset*, the RRset to sign/validate.  This can be a
    ``dns.rrset.RRset`` or a (``dns.name.Name``, ``dns.rdataset.Rdataset``)
    tuple.

    *rrsig*, a ``dns.rdata.Rdata``, the signature to validate, or the
    signature template used when signing.

    *origin*, a ``dns.name.Name`` or ``None``, the origin to use for relative
    names.

    Raises ``UnsupportedAlgorithm`` if the algorithm is recognized by
    dnspython but not implemented.
    """
    if isinstance(origin, str):
        origin = dns.name.from_text(origin, dns.name.root)
    signer = rrsig.signer
    if not signer.is_absolute():
        if origin is None:
            raise ValidationFailure('relative RR name without an origin specified')
        signer = signer.derelativize(origin)
    rrname, rdataset = _get_rrname_rdataset(rrset)
    data = b''
    data += rrsig.to_wire(origin=signer)[:18]
    data += rrsig.signer.to_digestable(signer)
    if not rrname.is_absolute():
        if origin is None:
            raise ValidationFailure('relative RR name without an origin specified')
        rrname = rrname.derelativize(origin)
    name_len = len(rrname)
    if rrname.is_wild() and rrsig.labels != name_len - 2:
        raise ValidationFailure('wild owner name has wrong label length')
    if name_len - 1 < rrsig.labels:
        raise ValidationFailure('owner name longer than RRSIG labels')
    elif rrsig.labels < name_len - 1:
        suffix = rrname.split(rrsig.labels + 1)[1]
        rrname = dns.name.from_text('*', suffix)
    rrnamebuf = rrname.to_digestable()
    rrfixed = struct.pack('!HHI', rdataset.rdtype, rdataset.rdclass, rrsig.original_ttl)
    rdatas = [rdata.to_digestable(origin) for rdata in rdataset]
    for rdata in sorted(rdatas):
        data += rrnamebuf
        data += rrfixed
        rrlen = struct.pack('!H', len(rdata))
        data += rrlen
        data += rdata
    return data