from __future__ import absolute_import, division, print_function
import json
import os
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def _ensure_alert_policy_is_absent(self):
    """
        Ensures that the alert policy is absent
        :return: (changed, None)
                 changed: A flag representing if anything is modified
        """
    changed = False
    p = self.module.params
    alert_policy_id = p.get('id')
    alert_policy_name = p.get('name')
    alias = p.get('alias')
    if not alert_policy_id and (not alert_policy_name):
        self.module.fail_json(msg='Either alert policy id or policy name is required')
    if not alert_policy_id and alert_policy_name:
        alert_policy_id = self._get_alert_policy_id(self.module, alert_policy_name)
    if alert_policy_id and alert_policy_id in self.policy_dict:
        changed = True
        if not self.module.check_mode:
            self._delete_alert_policy(alias, alert_policy_id)
    return (changed, None)