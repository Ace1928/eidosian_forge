from __future__ import absolute_import, division, print_function
import json
import os
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def _create_alert_policy(self):
    """
        Create an alert Policy using the CLC API.
        :return: response dictionary from the CLC API.
        """
    p = self.module.params
    alias = p['alias']
    email_list = p['alert_recipients']
    metric = p['metric']
    duration = p['duration']
    threshold = p['threshold']
    policy_name = p['name']
    arguments = json.dumps({'name': policy_name, 'actions': [{'action': 'email', 'settings': {'recipients': email_list}}], 'triggers': [{'metric': metric, 'duration': duration, 'threshold': threshold}]})
    try:
        result = self.clc.v2.API.Call('POST', '/v2/alertPolicies/%s' % alias, arguments)
    except APIFailedResponse as e:
        return self.module.fail_json(msg='Unable to create alert policy "{0}". {1}'.format(policy_name, str(e.response_text)))
    return result