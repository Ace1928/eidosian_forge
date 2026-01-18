from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def _node_has_migs(self, node=None):
    """just calls namespace_has_migs and
        if any namespace has migs returns true"""
    migs = 0
    self._update_cluster_namespace_list()
    for namespace in self._namespaces:
        if self._namespace_has_migs(namespace, node):
            migs += 1
    return migs != 0