from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def __populate_vm_host_list(self, group_name=None, cluster_obj=None, host_group=False):
    """
        Return all VM/Host names using given group name
        Args:
            group_name: group name
            cluster_obj: Cluster managed object
            host_group: True if we want only host name from group

        Returns: List of VM/Host names belonging to given group object

        """
    obj_name_list = []
    if group_name is None:
        group_name = self.__group_name
    if cluster_obj is None:
        cluster_obj = self.__cluster_obj
    if not all([group_name, cluster_obj]):
        return obj_name_list
    group = self.__group_obj
    if not host_group and isinstance(group, vim.cluster.VmGroup):
        obj_name_list = [vm.name for vm in group.vm]
    elif host_group and isinstance(group, vim.cluster.HostGroup):
        obj_name_list = [host.name for host in group.host]
    return obj_name_list