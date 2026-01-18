from __future__ import absolute_import, division, print_function
import os
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def _policy_exists(self, policy_name):
    """
        Check to see if an Anti Affinity Policy exists
        :param policy_name: name of the policy
        :return: boolean of if the policy exists
        """
    if policy_name in self.policy_dict:
        return self.policy_dict.get(policy_name)
    return False