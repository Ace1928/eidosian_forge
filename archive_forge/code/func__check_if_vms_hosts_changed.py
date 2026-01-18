from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def _check_if_vms_hosts_changed(self, group_name=None, cluster_obj=None, host_group=False):
    """
        Check if VMs/Hosts changed
        Args:
            group_name: Name of group
            cluster_obj: vim Cluster object
            host_group: True if we want to check hosts, else check vms

        Returns: Bool

        """
    if group_name is None:
        group_name = self._group_name
    if cluster_obj is None:
        cluster_obj = self._cluster_obj
    list_a = self._host_list if host_group else self._vm_list
    list_b = self._populate_vm_host_list(host_group=host_group)
    if set(list_a) == set(list_b):
        if self._operation != 'remove':
            return False
    return True