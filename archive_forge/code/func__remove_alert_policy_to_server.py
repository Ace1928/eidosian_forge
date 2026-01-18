from __future__ import absolute_import, division, print_function
import json
import os
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
@staticmethod
def _remove_alert_policy_to_server(clc, module, acct_alias, server_id, alert_policy_id):
    """
        remove the alert policy to the CLC server
        :param clc: the clc-sdk instance to use
        :param module: the AnsibleModule object
        :param acct_alias: the CLC account alias
        :param server_id: the CLC server id
        :param alert_policy_id: the alert policy id
        :return: result: The result from the CLC API call
        """
    result = None
    if not module.check_mode:
        try:
            result = clc.v2.API.Call('DELETE', 'servers/%s/%s/alertPolicies/%s' % (acct_alias, server_id, alert_policy_id))
        except APIFailedResponse as ex:
            module.fail_json(msg='Unable to remove alert policy from the server : "{0}". {1}'.format(server_id, str(ex.response_text)))
    return result