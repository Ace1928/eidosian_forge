from __future__ import absolute_import, division, print_function
from ast import literal_eval
from ansible.module_utils._text import to_text
from ansible.module_utils.common.validation import check_required_arguments
from ansible.module_utils.connection import ConnectionError
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
def _state_merged_overridden_replaced(self, want, have, state):
    """ The command generator when state is merged/overridden/replaced

        :rtype: A list
        :returns: the commands necessary to migrate the current configuration
                  to the desired configuration
        """
    add_commands = []
    del_commands = []
    commands = []
    add_requests = []
    del_requests = []
    requests = []
    have_dict = self._convert_config_list_to_dict(have)
    want_dict = self._convert_config_list_to_dict(want)
    for acl_type in ('ipv4', 'ipv6'):
        acl_type_add_commands = []
        acl_type_del_commands = []
        have_acl_names = set(have_dict.get(acl_type, {}).keys())
        want_acl_names = set(want_dict.get(acl_type, {}).keys())
        if state == 'overridden':
            for acl_name in have_acl_names.difference(want_acl_names):
                acl_type_del_commands.append({'name': acl_name})
                del_requests.append(self.get_delete_l3_acl_request(acl_type, acl_name))
        for acl_name in want_acl_names.intersection(have_acl_names):
            acl_add_command = {'name': acl_name}
            acl_del_command = {'name': acl_name}
            rule_add_commands = []
            rule_del_commands = []
            have_acl = have_dict[acl_type][acl_name]
            want_acl = want_dict[acl_type][acl_name]
            if not want_acl['remark']:
                if have_acl['remark'] and state in ('replaced', 'overridden'):
                    acl_del_command['remark'] = have_acl['remark']
                    del_requests.append(self.get_delete_l3_acl_remark_request(acl_type, acl_name))
            elif want_acl['remark'] != have_acl['remark']:
                acl_add_command['remark'] = want_acl['remark']
                add_requests.append(self.get_create_l3_acl_remark_request(acl_type, acl_name, want_acl['remark']))
            have_seq_nums = set(have_acl['rules'].keys())
            want_seq_nums = set(want_acl['rules'].keys())
            if state in ('replaced', 'overridden'):
                for seq_num in have_seq_nums.difference(want_seq_nums):
                    rule_del_commands.append({'sequence_num': seq_num})
                    del_requests.append(self.get_delete_l3_acl_rule_request(acl_type, acl_name, seq_num))
            for seq_num in want_seq_nums.intersection(have_seq_nums):
                if have_acl['rules'][seq_num] != want_acl['rules'][seq_num]:
                    if state == 'merged':
                        self._module.fail_json(msg='Cannot update existing sequence {0} of {1} ACL {2} with state merged. Please use state replaced or overridden.'.format(seq_num, acl_type, acl_name))
                    rule_del_commands.append({'sequence_num': seq_num})
                    del_requests.append(self.get_delete_l3_acl_rule_request(acl_type, acl_name, seq_num))
                    rule_add_commands.append(want_acl['rules'][seq_num])
                    add_requests.append(self.get_create_l3_acl_rule_request(acl_type, acl_name, seq_num, want_acl['rules'][seq_num]))
            for seq_num in want_seq_nums.difference(have_seq_nums):
                rule_add_commands.append(want_acl['rules'][seq_num])
                add_requests.append(self.get_create_l3_acl_rule_request(acl_type, acl_name, seq_num, want_acl['rules'][seq_num]))
            if rule_del_commands:
                acl_del_command['rules'] = rule_del_commands
            if rule_add_commands:
                acl_add_command['rules'] = rule_add_commands
            if acl_del_command.get('rules') or acl_del_command.get('remark'):
                acl_type_del_commands.append(acl_del_command)
            if acl_add_command.get('rules') or acl_add_command.get('remark'):
                acl_type_add_commands.append(acl_add_command)
        for acl_name in want_acl_names.difference(have_acl_names):
            acl_add_command = {'name': acl_name}
            add_requests.append(self.get_create_l3_acl_request(acl_type, acl_name))
            want_acl = want_dict[acl_type][acl_name]
            if want_acl['remark']:
                acl_add_command['remark'] = want_acl['remark']
                add_requests.append(self.get_create_l3_acl_remark_request(acl_type, acl_name, want_acl['remark']))
            want_seq_nums = set(want_acl['rules'].keys())
            if want_seq_nums:
                acl_add_command['rules'] = []
                for seq_num in want_seq_nums:
                    acl_add_command['rules'].append(want_acl['rules'][seq_num])
                    add_requests.append(self.get_create_l3_acl_rule_request(acl_type, acl_name, seq_num, want_acl['rules'][seq_num]))
            acl_type_add_commands.append(acl_add_command)
        if acl_type_del_commands:
            del_commands.append({'address_family': acl_type, 'acls': acl_type_del_commands})
        if acl_type_add_commands:
            add_commands.append({'address_family': acl_type, 'acls': acl_type_add_commands})
    if del_commands:
        commands = update_states(del_commands, 'deleted')
        requests = del_requests
    if add_commands:
        commands.extend(update_states(add_commands, state))
        requests.extend(add_requests)
    return (commands, requests)