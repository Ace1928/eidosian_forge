from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
class NetAppOntapEMSFilters:

    def __init__(self):
        self.argument_spec = netapp_utils.na_ontap_host_argument_spec()
        self.argument_spec.update(dict(state=dict(required=False, type='str', choices=['present', 'absent'], default='present'), name=dict(required=True, type='str'), rules=dict(type='list', elements='dict', options=dict(index=dict(required=True, type='int'), type=dict(required=True, type='str', choices=['include', 'exclude']), message_criteria=dict(type='dict', options=dict(severities=dict(required=False, type='str'), name_pattern=dict(required=False, type='str')))))))
        self.module = AnsibleModule(argument_spec=self.argument_spec, supports_check_mode=True)
        self.na_helper = NetAppModule(self.module)
        self.parameters = self.na_helper.check_and_set_parameters(self.module)
        self.rest_api = netapp_utils.OntapRestAPI(self.module)
        self.use_rest = self.rest_api.is_rest()
        if not self.use_rest:
            self.module.fail_json(msg='This module require REST with ONTAP 9.6 or higher')

    def get_ems_filter(self):
        api = 'support/ems/filters'
        params = {'name': self.parameters['name'], 'fields': 'rules'}
        record, error = rest_generic.get_one_record(self.rest_api, api, params)
        if error:
            self.module.fail_json(msg='Error fetching ems filter %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())
        return record

    def create_ems_filter(self):
        api = 'support/ems/filters'
        body = {'name': self.parameters['name']}
        if self.parameters.get('rules'):
            body['rules'] = self.na_helper.filter_out_none_entries(self.parameters['rules'])
        dummy, error = rest_generic.post_async(self.rest_api, api, body)
        if error:
            self.module.fail_json(msg='Error creating EMS filter %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())

    def delete_ems_filter(self):
        api = 'support/ems/filters'
        dummy, error = rest_generic.delete_async(self.rest_api, api, self.parameters['name'])
        if error:
            self.module.fail_json(msg='Error deleting EMS filter %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())

    def modify_ems_filter(self, desired_rules):
        post_api = 'support/ems/filters/%s/rules' % self.parameters['name']
        api = 'support/ems/filters'
        if desired_rules['patch_rules'] != []:
            patch_body = {'rules': desired_rules['patch_rules']}
            dummy, error = rest_generic.patch_async(self.rest_api, api, self.parameters['name'], patch_body)
            if error:
                self.module.fail_json(msg='Error modifying EMS filter %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())
        if desired_rules['post_rules'] != []:
            for rule in desired_rules['post_rules']:
                dummy, error = rest_generic.post_async(self.rest_api, post_api, rule)
                if error:
                    self.module.fail_json(msg='Error modifying EMS filter %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())

    def desired_ems_rules(self, current_rules):
        current_rules['rules'] = current_rules['rules'][:-1]
        if self.parameters.get('rules'):
            input_rules = self.na_helper.filter_out_none_entries(self.parameters['rules'])
            for i in range(len(input_rules)):
                input_rules[i]['message_criteria']['severities'] = input_rules[i]['message_criteria']['severities'].lower()
            matched_idx = []
            patch_rules = []
            post_rules = []
            for rule_dict in current_rules['rules']:
                for i in range(len(input_rules)):
                    if input_rules[i]['index'] == rule_dict['index']:
                        matched_idx.append(int(input_rules[i]['index']))
                        patch_rules.append(input_rules[i])
                        break
                else:
                    rule = {'index': rule_dict['index']}
                    rule['type'] = rule_dict.get('type')
                    if 'message_criteria' in rule_dict:
                        rule['message_criteria'] = {}
                        rule['message_criteria']['severities'] = rule_dict.get('message_criteria').get('severities')
                        rule['message_criteria']['name_pattern'] = rule_dict.get('message_criteria').get('name_pattern')
                    patch_rules.append(rule)
            for i in range(len(input_rules)):
                if int(input_rules[i]['index']) not in matched_idx:
                    post_rules.append(input_rules[i])
            desired_rules = {'patch_rules': patch_rules, 'post_rules': post_rules}
            return desired_rules
        return None

    def find_modify(self, current, desired_rules):
        if not current:
            return False
        if current.get('rules') is None or desired_rules is None:
            return False
        modify = False
        merge_rules = desired_rules['patch_rules'] + desired_rules['post_rules']
        if len(current.get('rules')) != len(merge_rules):
            return True
        for i in range(len(current['rules'])):
            if current['rules'][i]['index'] != merge_rules[i]['index'] or current['rules'][i]['type'] != merge_rules[i]['type']:
                return True
            else:
                if merge_rules[i].get('message_criteria') is None:
                    merge_rules[i]['message_criteria'] = {'severities': '*', 'name_pattern': '*'}
                elif merge_rules[i]['message_criteria'].get('severities') is None:
                    merge_rules[i]['message_criteria']['severities'] = '*'
                elif merge_rules[i]['message_criteria'].get('name_pattern') is None:
                    merge_rules[i]['message_criteria']['name_pattern'] = '*'
                if current['rules'][i].get('message_criteria').get('name_pattern') != merge_rules[i].get('message_criteria').get('name_pattern'):
                    return True
                if current['rules'][i].get('message_criteria').get('severities') != merge_rules[i].get('message_criteria').get('severities'):
                    return True
        return modify

    def apply(self):
        current = self.get_ems_filter()
        cd_action, modify = (None, False)
        cd_action = self.na_helper.get_cd_action(current, self.parameters)
        if cd_action is None and self.parameters['state'] == 'present':
            desired_rules = self.desired_ems_rules(current)
            modify = self.find_modify(current, desired_rules)
            if modify:
                self.na_helper.changed = True
        if self.na_helper.changed and (not self.module.check_mode):
            if cd_action == 'create':
                self.create_ems_filter()
            if cd_action == 'delete':
                self.delete_ems_filter()
            if modify:
                self.modify_ems_filter(desired_rules)
        result = netapp_utils.generate_result(self.na_helper.changed, cd_action, modify)
        self.module.exit_json(**result)