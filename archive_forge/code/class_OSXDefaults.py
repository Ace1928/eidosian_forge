from __future__ import absolute_import, division, print_function
from datetime import datetime
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import binary_type, text_type
class OSXDefaults(object):
    """ Class to manage Mac OS user defaults """

    def __init__(self, module):
        """ Initialize this module. Finds 'defaults' executable and preps the parameters """
        self.current_value = None
        self.module = module
        self.domain = module.params['domain']
        self.host = module.params['host']
        self.key = module.params['key']
        self.type = module.params['type']
        self.array_add = module.params['array_add']
        self.value = module.params['value']
        self.state = module.params['state']
        self.path = module.params['path']
        self.executable = self.module.get_bin_path('defaults', required=False, opt_dirs=self.path.split(':'))
        if not self.executable:
            raise OSXDefaultsException('Unable to locate defaults executable.')
        if self.state != 'absent':
            self.value = self._convert_type(self.type, self.value)

    @staticmethod
    def is_int(value):
        as_str = str(value)
        if as_str.startswith('-'):
            return as_str[1:].isdigit()
        else:
            return as_str.isdigit()

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

    def _host_args(self):
        """ Returns a normalized list of commandline arguments based on the "host" attribute """
        if self.host is None:
            return []
        elif self.host == 'currentHost':
            return ['-currentHost']
        else:
            return ['-host', self.host]

    def _base_command(self):
        """ Returns a list containing the "defaults" executable and any common base arguments """
        return [self.executable] + self._host_args()

    @staticmethod
    def _convert_defaults_str_to_list(value):
        """ Converts array output from defaults to an list """
        value = value.splitlines()
        value.pop(0)
        value.pop(-1)
        value = [re.sub('^ *"?|"?,? *$', '', x.replace('\\"', '"')) for x in value]
        return value

    def read(self):
        """ Reads value of this domain & key from defaults """
        rc, out, err = self.module.run_command(self._base_command() + ['read-type', self.domain, self.key])
        if rc == 1:
            return None
        if rc != 0:
            raise OSXDefaultsException('An error occurred while reading key type from defaults: %s' % err)
        data_type = out.strip().replace('Type is ', '')
        rc, out, err = self.module.run_command(self._base_command() + ['read', self.domain, self.key])
        out = out.strip()
        if rc != 0:
            raise OSXDefaultsException('An error occurred while reading key value from defaults: %s' % err)
        if data_type == 'array':
            out = self._convert_defaults_str_to_list(out)
        self.current_value = self._convert_type(data_type, out)

    def write(self):
        """ Writes value to this domain & key to defaults """
        if isinstance(self.value, bool):
            if self.value:
                value = 'TRUE'
            else:
                value = 'FALSE'
        elif isinstance(self.value, (int, float)):
            value = str(self.value)
        elif self.array_add and self.current_value is not None:
            value = list(set(self.value) - set(self.current_value))
        elif isinstance(self.value, datetime):
            value = self.value.strftime('%Y-%m-%d %H:%M:%S')
        else:
            value = self.value
        if self.type == 'array' and self.array_add:
            self.type = 'array-add'
        if not isinstance(value, list):
            value = [value]
        rc, out, err = self.module.run_command(self._base_command() + ['write', self.domain, self.key, '-' + self.type] + value, expand_user_and_vars=False)
        if rc != 0:
            raise OSXDefaultsException('An error occurred while writing value to defaults: %s' % err)

    def delete(self):
        """ Deletes defaults key from domain """
        rc, out, err = self.module.run_command(self._base_command() + ['delete', self.domain, self.key])
        if rc != 0:
            raise OSXDefaultsException('An error occurred while deleting key from defaults: %s' % err)
    ' Does the magic! :) '

    def run(self):
        self.read()
        if self.state == 'list':
            self.module.exit_json(key=self.key, value=self.current_value)
        if self.state == 'absent':
            if self.current_value is None:
                return False
            if self.module.check_mode:
                return True
            self.delete()
            return True
        value_type = type(self.value)
        if self.current_value is not None and (not isinstance(self.current_value, value_type)):
            raise OSXDefaultsException('Type mismatch. Type in defaults: %s' % type(self.current_value).__name__)
        if self.type == 'array' and self.current_value is not None and (not self.array_add) and (set(self.current_value) == set(self.value)):
            return False
        elif self.type == 'array' and self.current_value is not None and self.array_add and (len(list(set(self.value) - set(self.current_value))) == 0):
            return False
        elif self.current_value == self.value:
            return False
        if self.module.check_mode:
            return True
        self.write()
        return True