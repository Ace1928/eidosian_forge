from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def _cluster_has_migs(self):
    """calls node_has_migs for each node"""
    migs = 0
    for node in self._nodes:
        if self._node_has_migs(node):
            migs += 1
    if migs == 0:
        return False
    return True