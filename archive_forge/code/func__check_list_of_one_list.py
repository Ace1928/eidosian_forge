from __future__ import (absolute_import, division, print_function)
from ansible.errors import AnsibleError
from ansible.module_utils.six import string_types
from ansible.plugins.lookup import LookupBase
from ansible.utils.listify import listify_lookup_plugin_terms
def _check_list_of_one_list(self, term):
    if isinstance(term, list) and len(term) == 1:
        term = term[0]
        if isinstance(term, list):
            term = self._check_list_of_one_list(term)
    return term