from __future__ import absolute_import, division, print_function
import os
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def _delete_policy(self, p):
    """
        Delete an Anti Affinity Policy using the CLC API.
        :param p: datacenter to delete a policy from
        :return: none
        """
    try:
        policy = self.policy_dict[p['name']]
        policy.Delete()
    except CLCException as ex:
        self.module.fail_json(msg='Failed to delete anti affinity policy : {0}. {1}'.format(p['name'], ex.response_text))