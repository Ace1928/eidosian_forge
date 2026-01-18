from __future__ import absolute_import, division, print_function
import json
import re
from ansible_collections.community.rabbitmq.plugins.module_utils.version import LooseVersion as Version
from ansible.module_utils.basic import AnsibleModule
def _policy_check_by_name(self, policy):
    if not policy:
        return False
    policy_name = policy.split('\t')[1]
    return policy_name == self._name