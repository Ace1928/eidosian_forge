from __future__ import (absolute_import, division, print_function)
import itertools
from ansible.errors import AnsibleError
from ansible.plugins.lookup import LookupBase
from ansible.utils.listify import listify_lookup_plugin_terms
def _lookup_variables(self, terms):
    results = []
    for x in terms:
        intermediate = listify_lookup_plugin_terms(x, templar=self._templar)
        results.append(intermediate)
    return results