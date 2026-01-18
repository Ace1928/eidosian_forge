from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def _local_node_has_migs(self):
    return self._node_has_migs(None)