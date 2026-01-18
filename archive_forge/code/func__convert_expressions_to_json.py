from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
from ansible.module_utils.compat.version import LooseVersion
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def _convert_expressions_to_json(self, expressions):
    expression_type_values = ['character_string_included', 'any_character_string_included', 'character_string_not_included', 'result_is_true', 'result_is_false']
    expression_jsons = []
    for expression in expressions:
        expression_json = {}
        expression_json['expression'] = expression['expression']
        expression_type = zabbix_utils.helper_to_numeric_value(expression_type_values, expression['expression_type'])
        expression_json['expression_type'] = str(expression_type)
        if expression['expression_type'] == 'any_character_string_included':
            if expression['exp_delimiter']:
                expression_json['exp_delimiter'] = expression['exp_delimiter']
            else:
                expression_json['exp_delimiter'] = ','
        elif expression['exp_delimiter']:
            self._module.warn("A value of exp_delimiter will be ignored because expression_type is not 'any_character_string_included'.")
        case_sensitive = '0'
        if expression['case_sensitive']:
            case_sensitive = '1'
        expression_json['case_sensitive'] = case_sensitive
        expression_jsons.append(expression_json)
    return expression_jsons