from __future__ import absolute_import, division, print_function
import json
import re
from ansible_collections.community.rabbitmq.plugins.module_utils.version import LooseVersion as Version
from ansible.module_utils.basic import AnsibleModule
def _policy_check(self, policy, name_fno=1, apply_to_fno=2, pattern_fno=3, tags_fno=4, priority_fno=5):
    if not policy:
        return False
    policy_data = policy.split('\t')
    policy_name = policy_data[name_fno]
    apply_to = policy_data[apply_to_fno]
    pattern = policy_data[pattern_fno].replace('\\\\', '\\')
    try:
        tags = json.loads(policy_data[tags_fno])
    except json.decoder.JSONDecodeError:
        tags = policy_data[tags_fno]
    priority = policy_data[priority_fno]
    return policy_name == self._name and apply_to == self._apply_to and (tags == self._tags) and (priority == self._priority) and (pattern == self._pattern)