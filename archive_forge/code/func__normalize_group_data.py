from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def _normalize_group_data(self, group_obj):
    """
        Return human readable group spec
        Args:
            group_obj: Group object

        Returns: DRS group object fact

        """
    if not all([group_obj]):
        return {}
    if hasattr(group_obj, 'host'):
        return dict(group_name=group_obj.name, hosts=self._host_list, type='host')
    return dict(group_name=group_obj.name, vms=self._vm_list, type='vm')