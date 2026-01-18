from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def _is_min_cluster_size(self):
    """checks that all nodes in the cluster are returning the
        minimum cluster size specified in their statistics output"""
    sizes = set()
    for node in self._cluster_statistics:
        sizes.add(int(self._cluster_statistics[node]['cluster_size']))
    if len(sizes) > 1:
        return False
    if min(sizes) >= self.module.params['min_cluster_size']:
        return True
    return False