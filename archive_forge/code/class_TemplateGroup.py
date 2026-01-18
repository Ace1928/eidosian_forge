from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
class TemplateGroup(ZabbixBase):

    def create_template_group(self, group_names):
        try:
            group_add_list = []
            for group_name in group_names:
                result = self._zapi.templategroup.get({'filter': {'name': group_name}})
                if not result:
                    if self._module.check_mode:
                        self._module.exit_json(changed=True)
                    self._zapi.templategroup.create({'name': group_name})
                    group_add_list.append(group_name)
            return group_add_list
        except Exception as e:
            self._module.fail_json(msg='Failed to create template group(s): %s' % e)

    def delete_template_group(self, group_ids):
        try:
            if self._module.check_mode:
                self._module.exit_json(changed=True)
            self._zapi.templategroup.delete(group_ids)
        except Exception as e:
            self._module.fail_json(msg='Failed to delete template group(s), Exception: %s' % e)

    def get_group_ids(self, template_groups):
        group_ids = []
        group_list = self._zapi.templategroup.get({'output': 'extend', 'filter': {'name': template_groups}})
        for group in group_list:
            group_id = group['groupid']
            group_ids.append(group_id)
        return (group_ids, group_list)