from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def _cluster_migrates_allowed(self):
    """ensure all nodes have 'migrate_allowed' in their stats output"""
    for node in self._nodes:
        node_stats = self._info_cmd_helper('statistics', node)
        allowed = node_stats['migrate_allowed']
        if allowed == 'false':
            return False
    return True