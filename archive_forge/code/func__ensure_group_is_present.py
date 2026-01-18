from __future__ import absolute_import, division, print_function
import os
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def _ensure_group_is_present(self, group_name, parent_name, group_description):
    """
        Checks to see if a server group exists, creates it if it doesn't.
        :param group_name: the name of the group to validate/create
        :param parent_name: the name of the parent group for group_name
        :param group_description: a short description of the server group (used when creating)
        :return: (changed, group) -
            changed:  Boolean- whether a change was made,
            group:  A clc group object for the group
        """
    if not self.root_group:
        raise AssertionError('Implementation Error: Root Group not set')
    parent = parent_name if parent_name is not None else self.root_group.name
    description = group_description
    changed = False
    group = group_name
    parent_exists = self._group_exists(group_name=parent, parent_name=None)
    child_exists = self._group_exists(group_name=group_name, parent_name=parent)
    if parent_exists and child_exists:
        group, parent = self.group_dict[group_name]
        changed = False
    elif parent_exists and (not child_exists):
        if not self.module.check_mode:
            group = self._create_group(group=group, parent=parent, description=description)
        changed = True
    else:
        self.module.fail_json(msg='parent group: ' + parent + ' does not exist')
    return (changed, group)