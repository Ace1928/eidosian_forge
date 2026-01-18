from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.text.converters import to_native, to_text
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