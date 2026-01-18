from __future__ import absolute_import, division, print_function
import json
import os
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
@staticmethod
def _get_aa_policy_id_by_name(clc, module, alias, aa_policy_name):
    """
        retrieves the anti affinity policy id of the server based on the name of the policy
        :param clc: the clc-sdk instance to use
        :param module: the AnsibleModule object
        :param alias: the CLC account alias
        :param aa_policy_name: the anti affinity policy name
        :return: aa_policy_id: The anti affinity policy id
        """
    aa_policy_id = None
    try:
        aa_policies = clc.v2.API.Call(method='GET', url='antiAffinityPolicies/%s' % alias)
    except APIFailedResponse as ex:
        return module.fail_json(msg='Unable to fetch anti affinity policies from account alias : "{0}". {1}'.format(alias, str(ex.response_text)))
    for aa_policy in aa_policies.get('items'):
        if aa_policy.get('name') == aa_policy_name:
            if not aa_policy_id:
                aa_policy_id = aa_policy.get('id')
            else:
                return module.fail_json(msg='multiple anti affinity policies were found with policy name : %s' % aa_policy_name)
    if not aa_policy_id:
        module.fail_json(msg='No anti affinity policy was found with policy name : %s' % aa_policy_name)
    return aa_policy_id