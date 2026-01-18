from __future__ import absolute_import, division, print_function
import json
import os
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def _delete_alert_policy(self, alias, policy_id):
    """
        Delete an alert policy using the CLC API.
        :param alias : the account alias
        :param policy_id: the alert policy id
        :return: response dictionary from the CLC API.
        """
    try:
        result = self.clc.v2.API.Call('DELETE', '/v2/alertPolicies/%s/%s' % (alias, policy_id), None)
    except APIFailedResponse as e:
        return self.module.fail_json(msg='Unable to delete alert policy id "{0}". {1}'.format(policy_id, str(e.response_text)))
    return result