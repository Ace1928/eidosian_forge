import socket
import time
from typing import Any, Dict, List, Optional, Union
import dns._ddr
import dns.asyncbackend
import dns.asyncquery
import dns.exception
import dns.name
import dns.query
import dns.rdataclass
import dns.rdatatype
import dns.resolver  # lgtm[py/import-and-import-from]
from dns.resolver import NXDOMAIN, NoAnswer, NoRootSOA, NotAbsolute
Try to update the resolver's nameservers using Discovery of Designated
        Resolvers (DDR).  If successful, the resolver will subsequently use
        DNS-over-HTTPS or DNS-over-TLS for future queries.

        *lifetime*, a float, is the maximum time to spend attempting DDR.  The default
        is 5 seconds.

        If the SVCB query is successful and results in a non-empty list of nameservers,
        then the resolver's nameservers are set to the returned servers in priority
        order.

        The current implementation does not use any address hints from the SVCB record,
        nor does it resolve addresses for the SCVB target name, rather it assumes that
        the bootstrap nameserver will always be one of the addresses and uses it.
        A future revision to the code may offer fuller support.  The code verifies that
        the bootstrap nameserver is in the Subject Alternative Name field of the
        TLS certficate.
        