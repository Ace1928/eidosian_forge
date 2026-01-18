from __future__ import absolute_import, division, print_function
import os
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def _get_policies_for_datacenter(self, p):
    """
        Get the Policies for a datacenter by calling the CLC API.
        :param p: datacenter to get policies from
        :return: policies in the datacenter
        """
    response = {}
    policies = self.clc.v2.AntiAffinity.GetAll(location=p['location'])
    for policy in policies:
        response[policy.name] = policy
    return response