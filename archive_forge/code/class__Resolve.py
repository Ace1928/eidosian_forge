from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.text.converters import to_native, to_text
class _Resolve(object):

    def __init__(self, timeout=10, timeout_retries=3, servfail_retries=0):
        self.timeout = timeout
        self.timeout_retries = timeout_retries
        self.servfail_retries = servfail_retries
        self.default_resolver = dns.resolver.get_default_resolver()

    def _handle_reponse_errors(self, target, response, nameserver=None, query=None, accept_errors=None):
        rcode = response.rcode()
        if rcode == dns.rcode.NOERROR:
            return True
        if accept_errors and rcode in accept_errors:
            return True
        if rcode == dns.rcode.NXDOMAIN:
            raise dns.resolver.NXDOMAIN(qnames=[target], responses={target: response})
        msg = 'Error %s' % dns.rcode.to_text(rcode)
        if nameserver:
            msg = '%s while querying %s' % (msg, nameserver)
        if query:
            msg = '%s with query %s' % (msg, query)
        raise ResolverError(msg)

    def _handle_timeout(self, function, *args, **kwargs):
        retry = 0
        while True:
            try:
                return function(*args, **kwargs)
            except dns.exception.Timeout as exc:
                if retry >= self.timeout_retries:
                    raise exc
                retry += 1

    def _resolve(self, resolver, dnsname, handle_response_errors=False, **kwargs):
        retry = 0
        while True:
            try:
                response = self._handle_timeout(resolver.resolve, dnsname, lifetime=self.timeout, **kwargs)
            except AttributeError:
                resolver.search = False
                try:
                    response = self._handle_timeout(resolver.query, dnsname, lifetime=self.timeout, **kwargs)
                except TypeError:
                    resolver.lifetime = self.timeout
                    response = self._handle_timeout(resolver.query, dnsname, **kwargs)
            if response.response.rcode() == dns.rcode.SERVFAIL and retry < self.servfail_retries:
                retry += 1
                continue
            if handle_response_errors:
                self._handle_reponse_errors(dnsname, response.response, nameserver=resolver.nameservers)
            return response.rrset