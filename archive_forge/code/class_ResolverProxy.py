import re
import struct
import sys
import eventlet
from eventlet import patcher
from eventlet.green import _socket_nodns
from eventlet.green import os
from eventlet.green import time
from eventlet.green import select
from eventlet.green import ssl
class ResolverProxy:
    """Resolver class which can also use /etc/hosts

    Initialise with a HostsResolver instance in order for it to also
    use the hosts file.
    """

    def __init__(self, hosts_resolver=None, filename='/etc/resolv.conf'):
        """Initialise the resolver proxy

        :param hosts_resolver: An instance of HostsResolver to use.

        :param filename: The filename containing the resolver
           configuration.  The default value is correct for both UNIX
           and Windows, on Windows it will result in the configuration
           being read from the Windows registry.
        """
        self._hosts = hosts_resolver
        self._filename = filename
        self._cached_resolver = None

    @property
    def _resolver(self):
        if self._cached_resolver is None:
            self.clear()
        return self._cached_resolver

    @_resolver.setter
    def _resolver(self, value):
        self._cached_resolver = value

    def clear(self):
        self._resolver = dns.resolver.Resolver(filename=self._filename)
        self._resolver.cache = dns.resolver.LRUCache()

    def query(self, qname, rdtype=dns.rdatatype.A, rdclass=dns.rdataclass.IN, tcp=False, source=None, raise_on_no_answer=True, _hosts_rdtypes=(dns.rdatatype.A, dns.rdatatype.AAAA), use_network=True):
        """Query the resolver, using /etc/hosts if enabled.

        Behavior:
        1. if hosts is enabled and contains answer, return it now
        2. query nameservers for qname if use_network is True
        3. if qname did not contain dots, pretend it was top-level domain,
           query "foobar." and append to previous result
        """
        result = [None, None, 0]
        if qname is None:
            qname = '0.0.0.0'
        if isinstance(qname, str) or isinstance(qname, bytes):
            qname = dns.name.from_text(qname, None)

        def step(fun, *args, **kwargs):
            try:
                a = fun(*args, **kwargs)
            except Exception as e:
                result[1] = e
                return False
            if a.rrset is not None and len(a.rrset):
                if result[0] is None:
                    result[0] = a
                else:
                    result[0].rrset.union_update(a.rrset)
                result[2] += len(a.rrset)
            return True

        def end():
            if result[0] is not None:
                if raise_on_no_answer and result[2] == 0:
                    raise dns.resolver.NoAnswer
                return result[0]
            if result[1] is not None:
                if raise_on_no_answer or not isinstance(result[1], dns.resolver.NoAnswer):
                    raise result[1]
            raise dns.resolver.NXDOMAIN(qnames=(qname,))
        if self._hosts and rdclass == dns.rdataclass.IN and (rdtype in _hosts_rdtypes):
            if step(self._hosts.query, qname, rdtype, raise_on_no_answer=False):
                if result[0] is not None or result[1] is not None or (not use_network):
                    return end()
        step(self._resolver.query, qname, rdtype, rdclass, tcp, source, raise_on_no_answer=False)
        if len(qname) == 1:
            step(self._resolver.query, qname.concatenate(dns.name.root), rdtype, rdclass, tcp, source, raise_on_no_answer=False)
        return end()

    def getaliases(self, hostname):
        """Return a list of all the aliases of a given hostname"""
        if self._hosts:
            aliases = self._hosts.getaliases(hostname)
        else:
            aliases = []
        while True:
            try:
                ans = self._resolver.query(hostname, dns.rdatatype.CNAME)
            except (dns.resolver.NoAnswer, dns.resolver.NXDOMAIN):
                break
            else:
                aliases.extend((str(rr.target) for rr in ans.rrset))
                hostname = ans[0].target
        return aliases