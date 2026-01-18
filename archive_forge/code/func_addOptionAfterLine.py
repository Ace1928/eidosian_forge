from __future__ import absolute_import, division, print_function
import os
import re
import tempfile
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_bytes
def addOptionAfterLine(option, value, iface, lines, last_line_dict, iface_options, address_family):
    if option == 'method':
        changed = False
        for ln in lines:
            if ln.get('line_type', '') == 'iface' and ln.get('iface', '') == iface and (value != ln.get('params', {}).get('method', '')):
                if address_family is not None and ln.get('address_family') != address_family:
                    continue
                changed = True
                ln['line'] = re.sub(ln.get('params', {}).get('method', '') + '$', value, ln.get('line'))
                ln['params']['method'] = value
        return (changed, lines)
    last_line = last_line_dict['line']
    prefix_start = last_line.find(last_line.split()[0])
    suffix_start = last_line.rfind(last_line.split()[-1]) + len(last_line.split()[-1])
    prefix = last_line[:prefix_start]
    if len(iface_options) < 1:
        prefix += '    '
    line = prefix + '%s %s' % (option, value) + last_line[suffix_start:]
    option_dict = optionDict(line, iface, option, value, address_family)
    index = len(lines) - lines[::-1].index(last_line_dict)
    lines.insert(index, option_dict)
    return (True, lines)