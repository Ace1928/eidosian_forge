from __future__ import absolute_import, division, print_function
import json
import os
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def _ensure_alert_policy_is_updated(self, alert_policy):
    """
        Ensures the alert policy is updated if anything is changed in the alert policy configuration
        :param alert_policy: the target alert policy
        :return: (changed, policy)
                 changed: A flag representing if anything is modified
                 policy: the updated the alert policy
        """
    changed = False
    p = self.module.params
    alert_policy_id = alert_policy.get('id')
    email_list = p.get('alert_recipients')
    metric = p.get('metric')
    duration = p.get('duration')
    threshold = p.get('threshold')
    policy = alert_policy
    if metric and metric != str(alert_policy.get('triggers')[0].get('metric')) or (duration and duration != str(alert_policy.get('triggers')[0].get('duration'))) or (threshold and float(threshold) != float(alert_policy.get('triggers')[0].get('threshold'))):
        changed = True
    elif email_list:
        t_email_list = list(alert_policy.get('actions')[0].get('settings').get('recipients'))
        if set(email_list) != set(t_email_list):
            changed = True
    if changed and (not self.module.check_mode):
        policy = self._update_alert_policy(alert_policy_id)
    return (changed, policy)