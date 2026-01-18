from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.text.converters import to_native, to_text
class SimpleResolver(_Resolve):

    def __init__(self, timeout=10, timeout_retries=3, servfail_retries=0):
        super(SimpleResolver, self).__init__(timeout=timeout, timeout_retries=timeout_retries, servfail_retries=servfail_retries)

    def resolve(self, target, nxdomain_is_empty=True, server_addresses=None, **kwargs):
        dnsname = dns.name.from_unicode(to_text(target))
        resolver = self.default_resolver
        if server_addresses:
            resolver = dns.resolver.Resolver(configure=False)
            resolver.timeout = self.timeout
            resolver.nameservers = server_addresses
        resolver.use_edns(0, ednsflags=dns.flags.DO, payload=_EDNS_SIZE)
        try:
            return self._resolve(resolver, dnsname, handle_response_errors=True, **kwargs)
        except dns.resolver.NXDOMAIN:
            if nxdomain_is_empty:
                return None
            raise
        except dns.resolver.NoAnswer:
            return None

    def resolve_addresses(self, target, **kwargs):
        dnsname = dns.name.from_unicode(to_text(target))
        resolver = self.default_resolver
        result = []
        try:
            for data in self._resolve(resolver, dnsname, handle_response_errors=True, rdtype=dns.rdatatype.A, **kwargs):
                result.append(str(data))
        except dns.resolver.NoAnswer:
            pass
        try:
            for data in self._resolve(resolver, dnsname, handle_response_errors=True, rdtype=dns.rdatatype.AAAA, **kwargs):
                result.append(str(data))
        except dns.resolver.NoAnswer:
            pass
        return result