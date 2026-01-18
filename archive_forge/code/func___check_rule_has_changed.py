from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
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