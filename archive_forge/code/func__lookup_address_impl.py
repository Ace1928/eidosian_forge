from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.text.converters import to_native, to_text
def _lookup_address_impl(self, target, rdtype):
    try:
        answer = self._resolve(self.default_resolver, target, handle_response_errors=True, rdtype=rdtype)
        return [str(res) for res in answer]
    except dns.resolver.NoAnswer:
        return []