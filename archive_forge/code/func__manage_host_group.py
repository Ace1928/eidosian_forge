from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def _manage_host_group(self):
    if self._check_if_vms_hosts_changed(host_group=True):
        need_reconfigure = False
        group = vim.cluster.HostGroup()
        group.name = self._group_name
        group.host = self._group_obj.host or []
        for host in self._host_obj_list:
            if self._operation == 'edit' and host not in group.host:
                group.host.append(host)
                need_reconfigure = True
            if self._operation == 'remove' and host in group.host:
                group.host.remove(host)
                need_reconfigure = True
        group_spec = vim.cluster.GroupSpec(info=group, operation='edit')
        config_spec = vim.cluster.ConfigSpecEx(groupSpec=[group_spec])
        if not self.module.check_mode and need_reconfigure:
            task = self._cluster_obj.ReconfigureEx(config_spec, modify=True)
            self.changed, dummy = wait_for_task(task)
        self._set_result(group)
        if self.changed:
            self.message = 'Updated host group %s successfully' % self._group_name
        else:
            self.message = 'No update to host group %s' % self._group_name
    else:
        self.changed = False
        self.message = 'No update to host group %s' % self._group_name