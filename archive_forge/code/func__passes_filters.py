from __future__ import absolute_import, division, print_function
import json
from ansible.errors import AnsibleError, AnsibleParserError
from ansible.module_utils._text import to_native
from ansible.module_utils.six.moves.urllib.error import HTTPError, URLError
from ansible.module_utils.urls import Request
from ansible.plugins.inventory import (BaseInventoryPlugin, Cacheable,
from ..module_utils.vultr_v2 import VULTR_USER_AGENT
def _passes_filters(self, filters, variables, host, strict=False):
    if filters and isinstance(filters, list):
        for template in filters:
            try:
                if not self._compose(template, variables):
                    return False
            except Exception as e:
                if strict:
                    raise AnsibleError('Could not evaluate host filter {0} for {1}: {2}'.format(template, host, to_native(e)))
                return False
    return True