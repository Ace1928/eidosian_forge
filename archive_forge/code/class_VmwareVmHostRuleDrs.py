from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
class VmwareVmHostRuleDrs(PyVmomi):
    """
    Class to manage VM HOST DRS Rules
    """

    def __init__(self, module):
        """
        Doctring: Init
        """
        super(VmwareVmHostRuleDrs, self).__init__(module)
        self.__datacenter_name = module.params.get('datacenter', None)
        self.__datacenter_obj = None
        self.__cluster_name = module.params['cluster_name']
        self.__cluster_obj = None
        self.__vm_group_name = module.params.get('vm_group_name', None)
        self.__host_group_name = module.params.get('host_group_name', None)
        self.__rule_name = module.params['drs_rule_name']
        self.__enabled = module.params['enabled']
        self.__mandatory = module.params['mandatory']
        self.__affinity_rule = module.params['affinity_rule']
        self.__msg = 'Nothing to see here...'
        self.__result = dict()
        self.__changed = False
        if self.__datacenter_name is not None:
            self.__datacenter_obj = find_datacenter_by_name(self.content, self.__datacenter_name)
            if self.__datacenter_obj is None and module.check_mode is False:
                raise Exception("Datacenter '%s' not found" % self.__datacenter_name)
        self.__cluster_obj = find_cluster_by_name(content=self.content, cluster_name=self.__cluster_name, datacenter=self.__datacenter_obj)
        if self.__cluster_obj is None and module.check_mode is False:
            raise Exception("Cluster '%s' not found" % self.__cluster_name)

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

    def get_changed(self):
        """
        Returns if anything changed
        Args: none

        Returns: boolean
        """
        return self.__changed

    def __get_rule_key_by_name(self, cluster_obj=None, rule_name=None):
        """
        Function to get a specific VM-Host DRS rule key by name
        Args:
            rule_name: Name of rule
            cluster_obj: Cluster managed object

        Returns: Rule Object if found or None

        """
        if cluster_obj is None:
            cluster_obj = self.__cluster_obj
        if rule_name is None:
            rule_name = self.__rule_name
        if rule_name:
            rules_list = [rule for rule in cluster_obj.configuration.rule if rule.name == rule_name]
            if rules_list:
                return rules_list[0]
        return None

    def __normalize_vm_host_rule_spec(self, rule_obj, cluster_obj=None):
        """
        Return human readable rule spec
        Args:
            rule_obj: Rule managed object
            cluster_obj: Cluster managed object

        Returns: Dictionary with VM-Host DRS Rule info

        """
        if cluster_obj is None:
            cluster_obj = self.__cluster_obj
        if not all([rule_obj, cluster_obj]):
            return {}
        return dict(rule_key=rule_obj.key, rule_enabled=rule_obj.enabled, rule_name=rule_obj.name, rule_mandatory=rule_obj.mandatory, rule_uuid=rule_obj.ruleUuid, rule_vm_group_name=rule_obj.vmGroupName, rule_affine_host_group_name=rule_obj.affineHostGroupName, rule_anti_affine_host_group_name=rule_obj.antiAffineHostGroupName, rule_vms=self.__get_all_from_group(group_name=rule_obj.vmGroupName, cluster_obj=cluster_obj), rule_affine_hosts=self.__get_all_from_group(group_name=rule_obj.affineHostGroupName, cluster_obj=cluster_obj, host_group=True), rule_anti_affine_hosts=self.__get_all_from_group(group_name=rule_obj.antiAffineHostGroupName, cluster_obj=cluster_obj, host_group=True), rule_type='vm_host_rule')

    def __get_all_from_group(self, group_name=None, cluster_obj=None, host_group=False):
        """
        Return all VM / Host names using given group name
        Args:
            group_name: Rule name
            cluster_obj: Cluster managed object
            host_group: True if we want only host name from group

        Returns: List of VM-Host names belonging to given group object

        """
        obj_name_list = []
        if not all([group_name, cluster_obj]):
            return obj_name_list
        for group in cluster_obj.configurationEx.group:
            if group.name != group_name:
                continue
            if not host_group and isinstance(group, vim.cluster.VmGroup):
                obj_name_list = [vm.name for vm in group.vm]
                break
            if host_group and isinstance(group, vim.cluster.HostGroup):
                obj_name_list = [host.name for host in group.host]
                break
        return obj_name_list

    def __check_rule_has_changed(self, rule_obj, cluster_obj=None):
        """
        Function to check if the rule being edited has changed
        """
        if cluster_obj is None:
            cluster_obj = self.__cluster_obj
        existing_rule = self.__normalize_vm_host_rule_spec(rule_obj=rule_obj, cluster_obj=cluster_obj)
        if existing_rule['rule_enabled'] == self.__enabled and existing_rule['rule_mandatory'] == self.__mandatory and (existing_rule['rule_vm_group_name'] == self.__vm_group_name) and (existing_rule['rule_affine_host_group_name'] == self.__host_group_name or existing_rule['rule_anti_affine_host_group_name'] == self.__host_group_name):
            return False
        else:
            return True

    def create(self):
        """
        Function to create a host VM-Host DRS rule if rule does not exist
        """
        rule_obj = self.__get_rule_key_by_name(rule_name=self.__rule_name)
        if rule_obj:
            operation = 'edit'
            rule_changed = self.__check_rule_has_changed(rule_obj)
        else:
            operation = 'add'
        if operation == 'add' or (operation == 'edit' and rule_changed is True):
            rule = vim.cluster.VmHostRuleInfo()
            if rule_obj:
                rule.key = rule_obj.key
            rule.enabled = self.__enabled
            rule.mandatory = self.__mandatory
            rule.name = self.__rule_name
            if self.__affinity_rule:
                rule.affineHostGroupName = self.__host_group_name
            else:
                rule.antiAffineHostGroupName = self.__host_group_name
            rule.vmGroupName = self.__vm_group_name
            rule_spec = vim.cluster.RuleSpec(info=rule, operation=operation)
            config_spec = vim.cluster.ConfigSpecEx(rulesSpec=[rule_spec])
            if not self.module.check_mode:
                task = self.__cluster_obj.ReconfigureEx(config_spec, modify=True)
                wait_for_task(task)
            self.__changed = True
        rule_obj = self.__get_rule_key_by_name(rule_name=self.__rule_name)
        self.__result = self.__normalize_vm_host_rule_spec(rule_obj)
        if operation == 'edit':
            self.__msg = 'Updated DRS rule `%s` successfully' % self.__rule_name
        else:
            self.__msg = 'Created DRS rule `%s` successfully' % self.__rule_name

    def delete(self, rule_name=None):
        """
        Function to delete VM-Host DRS rule using name
        """
        if rule_name is None:
            rule_name = self.__rule_name
        rule_obj = self.__get_rule_key_by_name(rule_name=rule_name)
        if rule_obj is not None:
            rule_key = int(rule_obj.key)
            rule_spec = vim.cluster.RuleSpec(removeKey=rule_key, operation='remove')
            config_spec = vim.cluster.ConfigSpecEx(rulesSpec=[rule_spec])
            if not self.module.check_mode:
                task = self.__cluster_obj.ReconfigureEx(config_spec, modify=True)
                wait_for_task(task)
            self.__changed = True
        if self.__changed:
            self.__msg = 'Deleted DRS rule `%s` successfully' % self.__rule_name
        else:
            self.__msg = 'DRS Rule `%s` does not exists or already deleted' % self.__rule_name