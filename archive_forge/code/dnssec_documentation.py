from io import BytesIO
import struct
import time
import dns.exception
import dns.name
import dns.node
import dns.rdataset
import dns.rdata
import dns.rdatatype
import dns.rdataclass
from ._compat import string_types
Validate an RRset.

    *rrset* is the RRset to validate.  It can be a ``dns.rrset.RRset`` or
    a ``(dns.name.Name, dns.rdataset.Rdataset)`` tuple.

    *rrsigset* is the signature RRset to be validated.  It can be a
    ``dns.rrset.RRset`` or a ``(dns.name.Name, dns.rdataset.Rdataset)`` tuple.

    *keys* is the key dictionary, used to find the DNSKEY associated with
    a given name.  The dictionary is keyed by a ``dns.name.Name``, and has
    ``dns.node.Node`` or ``dns.rdataset.Rdataset`` values.

    *origin* is a ``dns.name.Name``, the origin to use for relative names.

    *now* is an ``int``, the time to use when validating the signatures,
    in seconds since the UNIX epoch.  The default is the current time.
    