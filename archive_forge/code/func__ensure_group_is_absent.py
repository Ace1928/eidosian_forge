from __future__ import absolute_import, division, print_function
import os
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def _ensure_group_is_absent(self, group_name, parent_name):
    """
        Ensure that group_name is absent by deleting it if necessary
        :param group_name: string - the name of the clc server group to delete
        :param parent_name: string - the name of the parent group for group_name
        :return: changed, group
        """
    changed = False
    group = []
    results = []
    if self._group_exists(group_name=group_name, parent_name=parent_name):
        if not self.module.check_mode:
            group.append(group_name)
            result = self._delete_group(group_name)
            results.append(result)
        changed = True
    return (changed, group, results)