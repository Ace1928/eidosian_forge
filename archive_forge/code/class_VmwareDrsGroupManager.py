from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
class VmwareDrsGroupManager(PyVmomi):
    """
    Class to manage DRS groups
    """

    def __init__(self, module, cluster_name, group_name, state, datacenter_name=None, vm_list=None, host_list=None):
        """
        Init
        """
        super(VmwareDrsGroupManager, self).__init__(module)
        self.__datacenter_name = datacenter_name
        self.__datacenter_obj = None
        self.__cluster_name = cluster_name
        self.__cluster_obj = None
        self.__group_name = group_name
        self.__group_obj = None
        self.__operation = None
        self.__vm_list = vm_list
        self.__vm_obj_list = []
        self.__host_list = host_list
        self.__host_obj_list = []
        self.__msg = 'Nothing to see here...'
        self.__result = dict()
        self.__changed = False
        self.__state = state
        if datacenter_name is not None:
            self.__datacenter_obj = find_datacenter_by_name(self.content, self.__datacenter_name)
            if self.__datacenter_obj is None:
                raise Exception("Datacenter '%s' not found" % self.__datacenter_name)
        self.__cluster_obj = find_cluster_by_name(content=self.content, cluster_name=self.__cluster_name, datacenter=self.__datacenter_obj)
        if self.__cluster_obj is None:
            if not module.check_mode:
                raise Exception("Cluster '%s' not found" % self.__cluster_name)
        else:
            self.__group_obj = self.__get_group_by_name()
            self.__set_result(self.__group_obj)
        if self.__state == 'present':
            if self.__group_obj:
                self.__operation = 'edit'
            else:
                self.__operation = 'add'
            if self.__vm_list is not None:
                self.__set_vm_obj_list(vm_list=self.__vm_list)
            if self.__host_list is not None:
                self.__set_host_obj_list(host_list=self.__host_list)
        else:
            self.__operation = 'remove'

    def get_msg(self):
        """
        Returns message for Ansible result
        Args: none

        Returns: string
        """
        return self.__msg

    def get_result(self):
        """
        Returns result for Ansible
        Args: none

        Returns: dict
        """
        return self.__result

    def __set_result(self, group_obj):
        """
        Creates result for successful run
        Args:
            group_obj: group object

        Returns: None

        """
        self.__result = dict()
        if self.__cluster_obj is not None and group_obj is not None:
            self.__result[self.__cluster_obj.name] = []
            self.__result[self.__cluster_obj.name].append(self.__normalize_group_data(group_obj))

    def get_changed(self):
        """
        Returns if anything changed
        Args: none

        Returns: boolean
        """
        return self.__changed

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

    def __set_host_obj_list(self, host_list=None):
        """
        Function populate host object list from list of hostnames
        Args:
            host_list: List of host names

        Returns: None

        """
        if host_list is None:
            host_list = self.__host_list
        if host_list is not None:
            for host in host_list:
                if not self.module.check_mode:
                    host_obj = self.find_hostsystem_by_name(host)
                    if host_obj is None:
                        raise Exception('ESXi host %s does not exist in cluster %s' % (host, self.__cluster_name))
                    self.__host_obj_list.append(host_obj)

    def __get_group_by_name(self, group_name=None, cluster_obj=None):
        """
        Function to get group by name
        Args:
            group_name: Name of group
            cluster_obj: vim Cluster object

        Returns: Group Object if found or None

        """
        if group_name is None:
            group_name = self.__group_name
        if cluster_obj is None:
            cluster_obj = self.__cluster_obj
        if self.module.check_mode and cluster_obj is None:
            return None
        for group in cluster_obj.configurationEx.group:
            if group.name == group_name:
                return group
        return None

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

    def __check_if_vms_hosts_changed(self, group_name=None, cluster_obj=None, host_group=False):
        """
        Function to check if VMs/Hosts changed
        Args:
            group_name: Name of group
            cluster_obj: vim Cluster object
            host_group: True if we want to check hosts, else check vms

        Returns: Bool

        """
        if group_name is None:
            group_name = self.__group_name
        if cluster_obj is None:
            cluster_obj = self.__cluster_obj
        list_a = self.__host_list if host_group else self.__vm_list
        list_b = self.__populate_vm_host_list(host_group=host_group)
        if set(list_a) == set(list_b):
            return False
        return True

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

    def __create_vm_group(self):
        if self.__operation == 'add' or (self.__operation == 'edit' and self.__check_if_vms_hosts_changed()):
            group = vim.cluster.VmGroup()
            group.name = self.__group_name
            group.vm = self.__vm_obj_list
            group_spec = vim.cluster.GroupSpec(info=group, operation=self.__operation)
            config_spec = vim.cluster.ConfigSpecEx(groupSpec=[group_spec])
            changed = True
            if not self.module.check_mode:
                task = self.__cluster_obj.ReconfigureEx(config_spec, modify=True)
                changed, result = wait_for_task(task)
            self.__set_result(group)
            self.__changed = changed
        if self.__operation == 'edit':
            self.__msg = 'Updated vm group %s successfully' % self.__group_name
        else:
            self.__msg = 'Created vm group %s successfully' % self.__group_name

    def __normalize_group_data(self, group_obj):
        """
        Return human readable group spec
        Args:
            group_obj: Group object

        Returns: DRS group object fact

        """
        if not all([group_obj]):
            return {}
        if hasattr(group_obj, 'host'):
            return dict(group_name=group_obj.name, hosts=self.__host_list, type='host')
        return dict(group_name=group_obj.name, vms=self.__vm_list, type='vm')

    def create_drs_group(self):
        """
        Function to create a DRS host/vm group
        """
        if self.__vm_list is None:
            self.__create_host_group()
        elif self.__host_list is None:
            self.__create_vm_group()
        else:
            raise Exception('Failed, no hosts or vms defined')

    def delete_drs_group(self):
        """
        Function to delete a DRS host/vm group
        """
        if self.__group_obj is not None:
            self.__changed = True
            if not self.module.check_mode:
                group_spec = vim.cluster.GroupSpec(removeKey=self.__group_name, operation=self.__operation)
                config_spec = vim.cluster.ConfigSpecEx(groupSpec=[group_spec])
                task = self.__cluster_obj.ReconfigureEx(config_spec, modify=True)
                wait_for_task(task)
        if self.__changed:
            self.__msg = 'Deleted group `%s` successfully' % self.__group_name
        else:
            self.__msg = 'DRS group `%s` does not exists or already deleted' % self.__group_name