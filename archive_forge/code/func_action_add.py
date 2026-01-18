from __future__ import absolute_import, division, print_function
import os
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def action_add(module, existing_snapshots):
    commands = list()
    command = 'show snapshot sections'
    sections = []
    body = execute_show_command(command, module)[0]
    if body:
        section_regex = '.*\\[(?P<section>\\S+)\\].*'
        split_body = body.split('\n\n')
        for section in split_body:
            temp = {}
            for line in section.splitlines():
                try:
                    match_section = re.match(section_regex, section, re.DOTALL)
                    temp['section'] = match_section.groupdict()['section']
                except (AttributeError, KeyError):
                    pass
                if 'show command' in line:
                    temp['show_command'] = line.split('show command: ')[1]
                elif 'row id' in line:
                    temp['row_id'] = line.split('row id: ')[1]
                elif 'key1' in line:
                    temp['element_key1'] = line.split('key1: ')[1]
                elif 'key2' in line:
                    temp['element_key2'] = line.split('key2: ')[1]
            if temp:
                sections.append(temp)
    proposed = {'section': module.params['section'], 'show_command': module.params['show_command'], 'row_id': module.params['row_id'], 'element_key1': module.params['element_key1'], 'element_key2': module.params['element_key2'] or '-'}
    if proposed not in sections:
        if module.params['element_key2']:
            commands.append('snapshot section add {0} "{1}" {2} {3} {4}'.format(module.params['section'], module.params['show_command'], module.params['row_id'], module.params['element_key1'], module.params['element_key2']))
        else:
            commands.append('snapshot section add {0} "{1}" {2} {3}'.format(module.params['section'], module.params['show_command'], module.params['row_id'], module.params['element_key1']))
    return commands