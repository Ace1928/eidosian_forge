from __future__ import absolute_import, division, print_function
from datetime import datetime
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import binary_type, text_type
@staticmethod
def _convert_type(data_type, value):
    """ Converts value to given type """
    if data_type == 'string':
        return str(value)
    elif data_type in ['bool', 'boolean']:
        if isinstance(value, (binary_type, text_type)):
            value = value.lower()
        if value in [True, 1, 'true', '1', 'yes']:
            return True
        elif value in [False, 0, 'false', '0', 'no']:
            return False
        raise OSXDefaultsException('Invalid boolean value: {0}'.format(repr(value)))
    elif data_type == 'date':
        try:
            return datetime.strptime(value.split('+')[0].strip(), '%Y-%m-%d %H:%M:%S')
        except ValueError:
            raise OSXDefaultsException('Invalid date value: {0}. Required format yyy-mm-dd hh:mm:ss.'.format(repr(value)))
    elif data_type in ['int', 'integer']:
        if not OSXDefaults.is_int(value):
            raise OSXDefaultsException('Invalid integer value: {0}'.format(repr(value)))
        return int(value)
    elif data_type == 'float':
        try:
            value = float(value)
        except ValueError:
            raise OSXDefaultsException('Invalid float value: {0}'.format(repr(value)))
        return value
    elif data_type == 'array':
        if not isinstance(value, list):
            raise OSXDefaultsException('Invalid value. Expected value to be an array')
        return value
    raise OSXDefaultsException('Type is not supported: {0}'.format(data_type))