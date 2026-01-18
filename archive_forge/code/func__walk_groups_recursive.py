from __future__ import absolute_import, division, print_function
import os
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def _walk_groups_recursive(self, parent_group, child_group):
    """
        Walk a parent-child tree of groups, starting with the provided child group
        :param parent_group: clc_sdk.Group - the parent group to start the walk
        :param child_group: clc_sdk.Group - the child group to start the walk
        :return: a dictionary of groups and parents
        """
    result = {str(child_group): (child_group, parent_group)}
    groups = child_group.Subgroups().groups
    if len(groups) > 0:
        for group in groups:
            if group.type != 'default':
                continue
            result.update(self._walk_groups_recursive(child_group, group))
    return result