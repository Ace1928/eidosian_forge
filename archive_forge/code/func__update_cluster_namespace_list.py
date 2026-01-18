from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def _update_cluster_namespace_list(self):
    """ make a unique list of namespaces
        TODO: does this work on a rolling namespace add/deletion?
        thankfully if it doesn't, we dont need this on builds >=4.3"""
    self._namespaces = set()
    for node in self._nodes:
        namespaces = self._info_cmd_helper('namespaces', node)
        for namespace in namespaces:
            self._namespaces.add(namespace)