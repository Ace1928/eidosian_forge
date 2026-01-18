from __future__ import absolute_import, division, print_function
import json
import os
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def _ensure_aa_policy_present(self, server, server_params):
    """
        ensures the server is updated with the provided anti affinity policy
        :param server: the CLC server object
        :param server_params: the dictionary of server parameters
        :return: (changed, group) -
            changed: Boolean whether a change was made
            result: The result from the CLC API call
        """
    changed = False
    acct_alias = self.clc.v2.Account.GetAlias()
    aa_policy_id = server_params.get('anti_affinity_policy_id')
    aa_policy_name = server_params.get('anti_affinity_policy_name')
    if not aa_policy_id and aa_policy_name:
        aa_policy_id = self._get_aa_policy_id_by_name(self.clc, self.module, acct_alias, aa_policy_name)
    current_aa_policy_id = self._get_aa_policy_id_of_server(self.clc, self.module, acct_alias, server.id)
    if aa_policy_id and aa_policy_id != current_aa_policy_id:
        self._modify_aa_policy(self.clc, self.module, acct_alias, server.id, aa_policy_id)
        changed = True
    return changed