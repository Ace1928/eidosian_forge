from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
class VmwareDrsGroupMemberManager(PyVmomi):
    """
    Class to manage DRS group members
    """

    def __init__(self, module):
        """
        Init
        """
        super(VmwareDrsGroupMemberManager, self).__init__(module)
        self._datacenter_name = module.params.get('datacenter')
        self._datacenter_obj = None
        self._cluster_name = module.params.get('cluster')
        self._cluster_obj = None
        self._group_name = module.params.get('group_name')
        self._group_obj = None
        self._operation = None
        self._vm_list = module.params.get('vms')
        self._vm_obj_list = []
        self._host_list = module.params.get('hosts')
        self._host_obj_list = []
        self.message = 'Nothing to see here...'
        self.result = dict()
        self.changed = False
        self._state = module.params.get('state')
        if self._datacenter_name is not None:
            self._datacenter_obj = self.find_datacenter_by_name(self._datacenter_name)
            if self._datacenter_obj is None:
                self.module.fail_json(msg="Datacenter '%s' not found" % self._datacenter_name)
        self._cluster_obj = self.find_cluster_by_name(self._cluster_name, self._datacenter_obj)
        if self._cluster_obj is None:
            self.module.fail_json(msg="Cluster '%s' not found" % self._cluster_name)
        self._group_obj = self._get_group_by_name()
        if self._group_obj is None:
            self.module.fail_json(msg='Cluster %s does not have a DRS group %s' % (self._cluster_name, self._group_name))
        self._set_result(self._group_obj)
        self._operation = 'edit' if self._state == 'present' else 'remove'
        if self._vm_list is not None:
            self._set_vm_obj_list(vm_list=self._vm_list)
        if self._host_list is not None:
            self._set_host_obj_list(host_list=self._host_list)

    def _set_result(self, group_obj):
        """
        Creates result for successful run
        Args:
            group_obj: group object

        Returns: None

        """
        self.result = dict()
        if self._cluster_obj is not None and group_obj is not None:
            self.result[self._cluster_obj.name] = []
            self.result[self._cluster_obj.name].append(self._normalize_group_data(group_obj))

    def _set_vm_obj_list(self, vm_list=None, cluster_obj=None):
        """
        Populate vm object list from list of vms
        Args:
            vm_list: List of vm names

        Returns: None

        """
        if vm_list is None:
            vm_list = self._vm_list
        if cluster_obj is None:
            cluster_obj = self._cluster_obj
        if vm_list is not None:
            for vm in vm_list:
                if self.module.check_mode is False:
                    vm_obj = find_vm_by_id(content=self.content, vm_id=vm, vm_id_type='vm_name', cluster=cluster_obj)
                    if vm_obj is None:
                        self.module.fail_json(msg='VM %s does not exist in cluster %s' % (vm, self._cluster_name))
                    self._vm_obj_list.append(vm_obj)

    def _set_host_obj_list(self, host_list=None):
        """
        Populate host object list from list of hostnames
        Args:
            host_list: List of host names

        Returns: None

        """
        if host_list is None:
            host_list = self._host_list
        if host_list is not None:
            for host in host_list:
                if self.module.check_mode is False:
                    host_obj = self.find_hostsystem_by_name(host)
                    if host_obj is None and self.module.check_mode is False:
                        self.module.fail_json(msg='ESXi host %s does not exist in cluster %s' % (host, self._cluster_name))
                    self._host_obj_list.append(host_obj)

    def _get_group_by_name(self, group_name=None, cluster_obj=None):
        """
        Get group by name
        Args:
            group_name: Name of group
            cluster_obj: vim Cluster object

        Returns: Group Object if found or None

        """
        if group_name is None:
            group_name = self._group_name
        if cluster_obj is None:
            cluster_obj = self._cluster_obj
        if self.module.check_mode and cluster_obj is None:
            return None
        for group in cluster_obj.configurationEx.group:
            if group.name == group_name:
                return group
        return None

    def _populate_vm_host_list(self, group_name=None, cluster_obj=None, host_group=False):
        """
        Return all VMs/Hosts names using given group name
        Args:
            group_name: group name
            cluster_obj: Cluster managed object
            host_group: True if we want only host name from group

        Returns: List of VMs/Hosts names belonging to given group object

        """
        obj_name_list = []
        if group_name is None:
            group_name = self._group_name
        if cluster_obj is None:
            cluster_obj = self._cluster_obj
        if not all([group_name, cluster_obj]):
            return obj_name_list
        group = self._group_obj
        if not host_group and isinstance(group, vim.cluster.VmGroup):
            obj_name_list = [vm.name for vm in group.vm]
        elif host_group and isinstance(group, vim.cluster.HostGroup):
            obj_name_list = [host.name for host in group.host]
        return obj_name_list

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

    def _manage_vm_group(self):
        if self._check_if_vms_hosts_changed():
            need_reconfigure = False
            group = vim.cluster.VmGroup()
            group.name = self._group_name
            group.vm = self._group_obj.vm or []
            for vm in self._vm_obj_list:
                if self._operation == 'edit' and vm not in group.vm:
                    group.vm.append(vm)
                    need_reconfigure = True
                if self._operation == 'remove' and vm in group.vm:
                    group.vm.remove(vm)
                    need_reconfigure = True
            group_spec = vim.cluster.GroupSpec(info=group, operation='edit')
            config_spec = vim.cluster.ConfigSpecEx(groupSpec=[group_spec])
            if not self.module.check_mode and need_reconfigure:
                task = self._cluster_obj.ReconfigureEx(config_spec, modify=True)
                self.changed, dummy = wait_for_task(task)
            self._set_result(group)
            if self.changed:
                self.message = 'Updated vm group %s successfully' % self._group_name
            else:
                self.message = 'No update to vm group %s' % self._group_name
        else:
            self.changed = False
            self.message = 'No update to vm group %s' % self._group_name

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

    def manage_drs_group_members(self):
        """
        Add a DRS host/vm group members
        """
        if self._vm_list is None:
            self._manage_host_group()
        elif self._host_list is None:
            self._manage_vm_group()
        else:
            self.module.fail_json(msg='Failed, no hosts or vms defined')