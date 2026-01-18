import socket
import sys
import time
import random
import dns.exception
import dns.flags
import dns.ipv4
import dns.ipv6
import dns.message
import dns.name
import dns.query
import dns.rcode
import dns.rdataclass
import dns.rdatatype
import dns.reversename
import dns.tsig
from ._compat import xrange, string_types
def _getnameinfo(sockaddr, flags=0):
    host = sockaddr[0]
    port = sockaddr[1]
    if len(sockaddr) == 4:
        scope = sockaddr[3]
        family = socket.AF_INET6
    else:
        scope = None
        family = socket.AF_INET
    tuples = _getaddrinfo(host, port, family, socket.SOCK_STREAM, socket.SOL_TCP, 0)
    if len(tuples) > 1:
        raise socket.error('sockaddr resolved to multiple addresses')
    addr = tuples[0][4][0]
    if flags & socket.NI_DGRAM:
        pname = 'udp'
    else:
        pname = 'tcp'
    qname = dns.reversename.from_address(addr)
    if flags & socket.NI_NUMERICHOST == 0:
        try:
            answer = _resolver.query(qname, 'PTR')
            hostname = answer.rrset[0].target.to_text(True)
        except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer):
            if flags & socket.NI_NAMEREQD:
                raise socket.gaierror(socket.EAI_NONAME)
            hostname = addr
            if scope is not None:
                hostname += '%' + str(scope)
    else:
        hostname = addr
        if scope is not None:
            hostname += '%' + str(scope)
    if flags & socket.NI_NUMERICSERV:
        service = str(port)
    else:
        service = socket.getservbyport(port, pname)
    return (hostname, service)