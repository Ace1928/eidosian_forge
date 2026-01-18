from __future__ import absolute_import, division, print_function
import json
import os
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def _get_alert_policy_id(self, module, alert_policy_name):
    """
        retrieves the alert policy id of the account based on the name of the policy
        :param module: the AnsibleModule object
        :param alert_policy_name: the alert policy name
        :return: alert_policy_id: The alert policy id
        """
    alert_policy_id = None
    for policy_id in self.policy_dict:
        if self.policy_dict.get(policy_id).get('name') == alert_policy_name:
            if not alert_policy_id:
                alert_policy_id = policy_id
            else:
                return module.fail_json(msg='multiple alert policies were found with policy name : %s' % alert_policy_name)
    return alert_policy_id