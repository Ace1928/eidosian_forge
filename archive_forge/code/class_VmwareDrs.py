from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
class VmwareDrs(PyVmomi):

    def __init__(self, module):
        super(VmwareDrs, self).__init__(module)
        self.vm_list = module.params['vms']
        self.cluster_name = module.params['cluster_name']
        self.rule_name = module.params['drs_rule_name']
        self.enabled = module.params['enabled']
        self.mandatory = module.params['mandatory']
        self.affinity_rule = module.params['affinity_rule']
        self.state = module.params['state']
        self.cluster_obj = find_cluster_by_name(content=self.content, cluster_name=self.cluster_name)
        if self.cluster_obj is None:
            self.module.fail_json(msg='Failed to find the cluster %s' % self.cluster_name)
        self.vm_obj_list = []
        if self.state == 'present':
            self.vm_obj_list = self.get_all_vms_info()

    def get_all_vms_info(self, vms_list=None):
        """
        Get all VM objects using name from given cluster
        Args:
            vms_list: List of VM names

        Returns: List of VM managed objects

        """
        vm_obj_list = []
        if vms_list is None:
            vms_list = self.vm_list
        for vm_name in vms_list:
            vm_obj = find_vm_by_id(content=self.content, vm_id=vm_name, vm_id_type='vm_name', cluster=self.cluster_obj)
            if vm_obj is None:
                self.module.fail_json(msg='Failed to find the virtual machine %s in the given cluster %s' % (vm_name, self.cluster_name))
            vm_obj_list.append(vm_obj)
        return vm_obj_list

    def get_rule_key_by_name(self, cluster_obj=None, rule_name=None):
        """
        Get a specific DRS rule key by name
        Args:
            rule_name: Name of rule
            cluster_obj: Cluster managed object

        Returns: Rule Object if found or None

        """
        if cluster_obj is None:
            cluster_obj = self.cluster_obj
        if rule_name:
            rules_list = [rule for rule in cluster_obj.configuration.rule if rule.name == rule_name]
            if rules_list:
                return rules_list[0]
        return None

    @staticmethod
    def normalize_rule_spec(rule_obj=None):
        """
        Return human readable rule spec
        Args:
            rule_obj: Rule managed object

        Returns: Dictionary with Rule info

        """
        if rule_obj is None:
            return {}
        return dict(rule_key=rule_obj.key, rule_enabled=rule_obj.enabled, rule_name=rule_obj.name, rule_mandatory=rule_obj.mandatory, rule_uuid=rule_obj.ruleUuid, rule_vms=[vm.name for vm in rule_obj.vm], rule_affinity=True if isinstance(rule_obj, vim.cluster.AffinityRuleSpec) else False)

    def create(self):
        """
        Create a DRS rule if rule does not exist
        """
        rule_obj = self.get_rule_key_by_name(rule_name=self.rule_name)
        if rule_obj is not None:
            existing_rule = self.normalize_rule_spec(rule_obj=rule_obj)
            if sorted(existing_rule['rule_vms']) == sorted(self.vm_list) and existing_rule['rule_enabled'] == self.enabled and (existing_rule['rule_mandatory'] == self.mandatory) and (existing_rule['rule_affinity'] == self.affinity_rule):
                self.module.exit_json(changed=False, result=existing_rule, msg='Rule already exists with the same configuration')
            return self.update_rule_spec(rule_obj)
        return self.create_rule_spec()

    def create_rule_spec(self):
        """
        Create DRS rule
        """
        changed = False
        result = None
        if self.affinity_rule:
            rule = vim.cluster.AffinityRuleSpec()
        else:
            rule = vim.cluster.AntiAffinityRuleSpec()
        rule.vm = self.vm_obj_list
        rule.enabled = self.enabled
        rule.mandatory = self.mandatory
        rule.name = self.rule_name
        rule_spec = vim.cluster.RuleSpec(info=rule, operation='add')
        config_spec = vim.cluster.ConfigSpecEx(rulesSpec=[rule_spec])
        try:
            if not self.module.check_mode:
                task = self.cluster_obj.ReconfigureEx(config_spec, modify=True)
                changed, result = wait_for_task(task)
        except vmodl.fault.InvalidRequest as e:
            result = to_native(e.msg)
        except Exception as e:
            result = to_native(e)
        if changed:
            rule_obj = self.get_rule_key_by_name(rule_name=self.rule_name)
            result = self.normalize_rule_spec(rule_obj)
        if self.module.check_mode:
            changed = True
            result = dict(rule_key='', rule_enabled=rule.enabled, rule_name=self.rule_name, rule_mandatory=rule.mandatory, rule_uuid='', rule_vms=[vm.name for vm in rule.vm], rule_affinity=self.affinity_rule)
        return (changed, result)

    def update_rule_spec(self, rule_obj=None):
        """
        Update DRS rule
        """
        changed = False
        result = None
        rule_obj.vm = self.vm_obj_list
        if rule_obj.mandatory != self.mandatory:
            rule_obj.mandatory = self.mandatory
        if rule_obj.enabled != self.enabled:
            rule_obj.enabled = self.enabled
        rule_spec = vim.cluster.RuleSpec(info=rule_obj, operation='edit')
        config_spec = vim.cluster.ConfigSpec(rulesSpec=[rule_spec])
        try:
            if not self.module.check_mode:
                task = self.cluster_obj.ReconfigureCluster_Task(config_spec, modify=True)
                changed, result = wait_for_task(task)
            else:
                changed = True
        except vmodl.fault.InvalidRequest as e:
            result = to_native(e.msg)
        except Exception as e:
            result = to_native(e)
        if changed:
            rule_obj = self.get_rule_key_by_name(rule_name=self.rule_name)
            result = self.normalize_rule_spec(rule_obj)
        return (changed, result)

    def delete(self, rule_name=None):
        """
        Delete DRS rule using name
        """
        changed = False
        if rule_name is None:
            rule_name = self.rule_name
        rule = self.get_rule_key_by_name(rule_name=rule_name)
        if rule is not None:
            rule_key = int(rule.key)
            rule_spec = vim.cluster.RuleSpec(removeKey=rule_key, operation='remove')
            config_spec = vim.cluster.ConfigSpecEx(rulesSpec=[rule_spec])
            try:
                if not self.module.check_mode:
                    task = self.cluster_obj.ReconfigureEx(config_spec, modify=True)
                    changed, result = wait_for_task(task)
                else:
                    changed = True
                    result = 'Rule %s will be deleted' % self.rule_name
            except vmodl.fault.InvalidRequest as e:
                result = to_native(e.msg)
            except Exception as e:
                result = to_native(e)
        else:
            result = 'No rule named %s exists' % self.rule_name
        return (changed, result)