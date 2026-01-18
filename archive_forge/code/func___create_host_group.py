from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def __create_host_group(self):
    if self.__operation == 'add' or (self.__operation == 'edit' and self.__check_if_vms_hosts_changed(host_group=True)):
        group = vim.cluster.HostGroup()
        group.name = self.__group_name
        group.host = self.__host_obj_list
        group_spec = vim.cluster.GroupSpec(info=group, operation=self.__operation)
        config_spec = vim.cluster.ConfigSpecEx(groupSpec=[group_spec])
        changed = True
        if not self.module.check_mode:
            task = self.__cluster_obj.ReconfigureEx(config_spec, modify=True)
            changed, result = wait_for_task(task)
        self.__set_result(group)
        self.__changed = changed
    if self.__operation == 'edit':
        self.__msg = 'Updated host group %s successfully' % self.__group_name
    else:
        self.__msg = 'Created host group %s successfully' % self.__group_name