from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.text.converters import to_native, to_text
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