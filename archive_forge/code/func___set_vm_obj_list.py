from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def __set_vm_obj_list(self, vm_list=None, cluster_obj=None):
    """
        Function populate vm object list from list of vms
        Args:
            vm_list: List of vm names

        Returns: None

        """
    if vm_list is None:
        vm_list = self.__vm_list
    if cluster_obj is None:
        cluster_obj = self.__cluster_obj
    if vm_list is not None:
        for vm in vm_list:
            if not self.module.check_mode:
                vm_obj = find_vm_by_id(content=self.content, vm_id=vm, vm_id_type='vm_name', cluster=cluster_obj)
                if vm_obj is None:
                    raise Exception('VM %s does not exist in cluster %s' % (vm, self.__cluster_name))
                self.__vm_obj_list.append(vm_obj)