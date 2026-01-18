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
def _getaddrinfo(host=None, service=None, family=socket.AF_UNSPEC, socktype=0, proto=0, flags=0):
    if flags & (socket.AI_ADDRCONFIG | socket.AI_V4MAPPED) != 0:
        raise NotImplementedError
    if host is None and service is None:
        raise socket.gaierror(socket.EAI_NONAME)
    v6addrs = []
    v4addrs = []
    canonical_name = None
    try:
        if host is None:
            canonical_name = 'localhost'
            if flags & socket.AI_PASSIVE != 0:
                v6addrs.append('::')
                v4addrs.append('0.0.0.0')
            else:
                v6addrs.append('::1')
                v4addrs.append('127.0.0.1')
        else:
            parts = host.split('%')
            if len(parts) == 2:
                ahost = parts[0]
            else:
                ahost = host
            addr = dns.ipv6.inet_aton(ahost)
            v6addrs.append(host)
            canonical_name = host
    except Exception:
        try:
            addr = dns.ipv4.inet_aton(host)
            v4addrs.append(host)
            canonical_name = host
        except Exception:
            if flags & socket.AI_NUMERICHOST == 0:
                try:
                    if family == socket.AF_INET6 or family == socket.AF_UNSPEC:
                        v6 = _resolver.query(host, dns.rdatatype.AAAA, raise_on_no_answer=False)
                        host = v6.qname
                        canonical_name = v6.canonical_name.to_text(True)
                        if v6.rrset is not None:
                            for rdata in v6.rrset:
                                v6addrs.append(rdata.address)
                    if family == socket.AF_INET or family == socket.AF_UNSPEC:
                        v4 = _resolver.query(host, dns.rdatatype.A, raise_on_no_answer=False)
                        host = v4.qname
                        canonical_name = v4.canonical_name.to_text(True)
                        if v4.rrset is not None:
                            for rdata in v4.rrset:
                                v4addrs.append(rdata.address)
                except dns.resolver.NXDOMAIN:
                    raise socket.gaierror(socket.EAI_NONAME)
                except Exception:
                    raise socket.gaierror(socket.EAI_SYSTEM)
    port = None
    try:
        if service is None:
            port = 0
        else:
            port = int(service)
    except Exception:
        if flags & socket.AI_NUMERICSERV == 0:
            try:
                port = socket.getservbyname(service)
            except Exception:
                pass
    if port is None:
        raise socket.gaierror(socket.EAI_NONAME)
    tuples = []
    if socktype == 0:
        socktypes = [socket.SOCK_DGRAM, socket.SOCK_STREAM]
    else:
        socktypes = [socktype]
    if flags & socket.AI_CANONNAME != 0:
        cname = canonical_name
    else:
        cname = ''
    if family == socket.AF_INET6 or family == socket.AF_UNSPEC:
        for addr in v6addrs:
            for socktype in socktypes:
                for proto in _protocols_for_socktype[socktype]:
                    tuples.append((socket.AF_INET6, socktype, proto, cname, (addr, port, 0, 0)))
    if family == socket.AF_INET or family == socket.AF_UNSPEC:
        for addr in v4addrs:
            for socktype in socktypes:
                for proto in _protocols_for_socktype[socktype]:
                    tuples.append((socket.AF_INET, socktype, proto, cname, (addr, port)))
    if len(tuples) == 0:
        raise socket.gaierror(socket.EAI_NONAME)
    return tuples