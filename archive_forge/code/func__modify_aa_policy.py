from __future__ import absolute_import, division, print_function
import json
import os
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
@staticmethod
def _modify_aa_policy(clc, module, acct_alias, server_id, aa_policy_id):
    """
        modifies the anti affinity policy of the CLC server
        :param clc: the clc-sdk instance to use
        :param module: the AnsibleModule object
        :param acct_alias: the CLC account alias
        :param server_id: the CLC server id
        :param aa_policy_id: the anti affinity policy id
        :return: result: The result from the CLC API call
        """
    result = None
    if not module.check_mode:
        try:
            result = clc.v2.API.Call('PUT', 'servers/%s/%s/antiAffinityPolicy' % (acct_alias, server_id), json.dumps({'id': aa_policy_id}))
        except APIFailedResponse as ex:
            module.fail_json(msg='Unable to modify anti affinity policy to server : "{0}". {1}'.format(server_id, str(ex.response_text)))
    return result