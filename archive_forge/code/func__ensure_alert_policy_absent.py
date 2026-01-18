from __future__ import absolute_import, division, print_function
import json
import os
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def _ensure_alert_policy_absent(self, server, server_params):
    """
        ensures the alert policy is removed from the server
        :param server: the CLC server object
        :param server_params: the dictionary of server parameters
        :return: (changed, group) -
            changed: Boolean whether a change was made
            result: The result from the CLC API call
        """
    changed = False
    acct_alias = self.clc.v2.Account.GetAlias()
    alert_policy_id = server_params.get('alert_policy_id')
    alert_policy_name = server_params.get('alert_policy_name')
    if not alert_policy_id and alert_policy_name:
        alert_policy_id = self._get_alert_policy_id_by_name(self.clc, self.module, acct_alias, alert_policy_name)
    if alert_policy_id and self._alert_policy_exists(server, alert_policy_id):
        self._remove_alert_policy_to_server(self.clc, self.module, acct_alias, server.id, alert_policy_id)
        changed = True
    return changed