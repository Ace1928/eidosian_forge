from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.netscaler.netscaler import (
def action_identical(client, module, csaction_proxy):
    if len(diff_list(client, module, csaction_proxy)) == 0:
        return True
    else:
        return False