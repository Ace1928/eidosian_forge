from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.text.converters import to_native, to_text
class ResolveDirectlyFromNameServers(_Resolve):

    def __init__(self, timeout=10, timeout_retries=3, servfail_retries=0, always_ask_default_resolver=True, server_addresses=None):
        super(ResolveDirectlyFromNameServers, self).__init__(timeout=timeout, timeout_retries=timeout_retries, servfail_retries=servfail_retries)
        self.cache = {}
        self.default_nameservers = self.default_resolver.nameservers if server_addresses is None else server_addresses
        self.always_ask_default_resolver = always_ask_default_resolver

    def _lookup_ns_names(self, target, nameservers=None, nameserver_ips=None):
        if self.always_ask_default_resolver:
            nameservers = None
            nameserver_ips = self.default_nameservers
        if nameservers is None and nameserver_ips is None:
            nameserver_ips = self.default_nameservers
        if not nameserver_ips and nameservers:
            nameserver_ips = self._lookup_address(nameservers[0])
        if not nameserver_ips:
            raise ResolverError('Have neither nameservers nor nameserver IPs')
        query = dns.message.make_query(target, dns.rdatatype.NS)
        retry = 0
        while True:
            response = self._handle_timeout(dns.query.udp, query, nameserver_ips[0], timeout=self.timeout)
            if response.rcode() == dns.rcode.SERVFAIL and retry < self.servfail_retries:
                retry += 1
                continue
            break
        self._handle_reponse_errors(target, response, nameserver=nameserver_ips[0], query='get NS for "%s"' % target, accept_errors=[dns.rcode.NXDOMAIN])
        cname = None
        for rrset in response.answer:
            if rrset.rdtype == dns.rdatatype.CNAME:
                cname = dns.name.from_text(to_text(rrset[0]))
        new_nameservers = []
        rrsets = list(response.authority)
        rrsets.extend(response.answer)
        for rrset in rrsets:
            if rrset.rdtype == dns.rdatatype.SOA:
                return (None, cname)
            if rrset.rdtype == dns.rdatatype.NS:
                new_nameservers.extend((str(ns_record.target) for ns_record in rrset))
        return (sorted(set(new_nameservers)) if new_nameservers else None, cname)

    def _lookup_address_impl(self, target, rdtype):
        try:
            answer = self._resolve(self.default_resolver, target, handle_response_errors=True, rdtype=rdtype)
            return [str(res) for res in answer]
        except dns.resolver.NoAnswer:
            return []

    def _lookup_address(self, target):
        result = self.cache.get((target, 'addr'))
        if not result:
            result = self._lookup_address_impl(target, dns.rdatatype.A)
            result.extend(self._lookup_address_impl(target, dns.rdatatype.AAAA))
            self.cache[target, 'addr'] = result
        return result

    def _do_lookup_ns(self, target):
        nameserver_ips = self.default_nameservers
        nameservers = None
        for i in range(2, len(target.labels) + 1):
            target_part = target.split(i)[1]
            _nameservers = self.cache.get((str(target_part), 'ns'))
            if _nameservers is None:
                nameserver_names, cname = self._lookup_ns_names(target_part, nameservers=nameservers, nameserver_ips=nameserver_ips)
                if nameserver_names is not None:
                    nameservers = nameserver_names
                self.cache[str(target_part), 'ns'] = nameservers
                self.cache[str(target_part), 'cname'] = cname
            else:
                nameservers = _nameservers
            nameserver_ips = None
        return nameservers

    def _lookup_ns(self, target):
        result = self.cache.get((str(target), 'ns'))
        if not result:
            result = self._do_lookup_ns(target)
            self.cache[str(target), 'ns'] = result
        return result

    def _get_resolver(self, dnsname, nameservers):
        cache_index = ('|'.join([str(dnsname)] + sorted(nameservers)), 'resolver')
        resolver = self.cache.get(cache_index)
        if resolver is None:
            resolver = dns.resolver.Resolver(configure=False)
            resolver.use_edns(0, ednsflags=dns.flags.DO, payload=_EDNS_SIZE)
            resolver.timeout = self.timeout
            nameserver_ips = set()
            for nameserver in nameservers:
                nameserver_ips.update(self._lookup_address(nameserver))
            resolver.nameservers = sorted(nameserver_ips)
            self.cache[cache_index] = resolver
        return resolver

    def resolve_nameservers(self, target, resolve_addresses=False):
        nameservers = self._lookup_ns(dns.name.from_unicode(to_text(target)))
        if resolve_addresses:
            nameserver_ips = set()
            for nameserver in nameservers or []:
                nameserver_ips.update(self._lookup_address(nameserver))
            nameservers = list(nameserver_ips)
        return sorted(nameservers or [])

    def resolve(self, target, nxdomain_is_empty=True, **kwargs):
        dnsname = dns.name.from_unicode(to_text(target))
        loop_catcher = set()
        while True:
            try:
                nameservers = self._lookup_ns(dnsname)
            except dns.resolver.NXDOMAIN:
                if nxdomain_is_empty:
                    return {}
                raise
            cname = self.cache.get((str(dnsname), 'cname'))
            if cname is None:
                break
            dnsname = cname
            if dnsname in loop_catcher:
                raise ResolverError('Found CNAME loop starting at {0}'.format(target))
            loop_catcher.add(dnsname)
        results = {}
        for nameserver in nameservers or []:
            results[nameserver] = None
            resolver = self._get_resolver(dnsname, [nameserver])
            try:
                results[nameserver] = self._resolve(resolver, dnsname, handle_response_errors=True, **kwargs)
            except dns.resolver.NoAnswer:
                pass
            except dns.resolver.NXDOMAIN:
                if nxdomain_is_empty:
                    results[nameserver] = []
                else:
                    raise
        return results