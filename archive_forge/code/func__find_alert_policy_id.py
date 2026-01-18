from __future__ import absolute_import, division, print_function
import json
import os
import time
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
@staticmethod
def _find_alert_policy_id(clc, module):
    """
        Validate if the alert policy exist for the given name and throw error if not
        :param clc: the clc-sdk instance
        :param module: the module to validate
        :return: alert_policy_id: the alert policy id of the given name.
        """
    alert_policy_id = module.params.get('alert_policy_id')
    alert_policy_name = module.params.get('alert_policy_name')
    if not alert_policy_id and alert_policy_name:
        alias = module.params.get('alias')
        alert_policy_id = ClcServer._get_alert_policy_id_by_name(clc=clc, module=module, alias=alias, alert_policy_name=alert_policy_name)
        if not alert_policy_id:
            module.fail_json(msg='No alert policy exist with name : %s' % alert_policy_name)
    return alert_policy_id