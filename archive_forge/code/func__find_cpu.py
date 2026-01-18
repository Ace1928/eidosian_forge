from __future__ import absolute_import, division, print_function
import json
import os
import time
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
@staticmethod
def _find_cpu(clc, module):
    """
        Find or validate the CPU value by calling the CLC API
        :param clc: clc-sdk instance to use
        :param module: module to validate
        :return: Int value for CPU
        """
    cpu = module.params.get('cpu')
    group_id = module.params.get('group_id')
    alias = module.params.get('alias')
    state = module.params.get('state')
    if not cpu and state == 'present':
        group = clc.v2.Group(id=group_id, alias=alias)
        if group.Defaults('cpu'):
            cpu = group.Defaults('cpu')
        else:
            module.fail_json(msg=str("Can't determine a default cpu value. Please provide a value for cpu."))
    return cpu