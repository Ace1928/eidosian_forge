from __future__ import absolute_import, division, print_function
import os
import platform
import re
import tempfile
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import string_types
from ansible.module_utils.parsing.convert_bool import BOOLEANS_FALSE, BOOLEANS_TRUE
from ansible.module_utils._text import to_native
class SysctlModule(object):
    LANG_ENV = {'LANG': 'C', 'LC_ALL': 'C', 'LC_MESSAGES': 'C'}

    def __init__(self, module):
        self.module = module
        self.args = self.module.params
        self.sysctl_cmd = self.module.get_bin_path('sysctl', required=True)
        self.sysctl_file = self.args['sysctl_file']
        self.proc_value = None
        self.file_value = None
        self.file_lines = []
        self.file_values = {}
        self.changed = False
        self.set_proc = False
        self.write_file = False
        self.process()

    def process(self):
        self.platform = platform.system().lower()
        self.args['name'] = self.args['name'].strip()
        self.args['value'] = self._parse_value(self.args['value'])
        thisname = self.args['name']
        self.proc_value = self.get_token_curr_value(thisname)
        self.read_sysctl_file()
        if thisname not in self.file_values:
            self.file_values[thisname] = None
        self.fix_lines()
        if self.file_values[thisname] is None and self.args['state'] == 'present':
            self.changed = True
            self.write_file = True
        elif self.file_values[thisname] is None and self.args['state'] == 'absent':
            self.changed = False
        elif self.file_values[thisname] and self.args['state'] == 'absent':
            self.changed = True
            self.write_file = True
        elif self.file_values[thisname] != self.args['value']:
            self.changed = True
            self.write_file = True
        elif self.args['reload']:
            if self.proc_value is None:
                self.changed = True
            elif not self._values_is_equal(self.proc_value, self.args['value']):
                self.changed = True
        if self.args['sysctl_set'] and self.args['state'] == 'present':
            if self.proc_value is None:
                self.changed = True
            elif not self._values_is_equal(self.proc_value, self.args['value']):
                self.changed = True
                self.set_proc = True
        if not self.module.check_mode:
            if self.set_proc:
                self.set_token_value(self.args['name'], self.args['value'])
            if self.write_file:
                self.write_sysctl()
            if self.changed and self.args['reload']:
                self.reload_sysctl()

    def _values_is_equal(self, a, b):
        """Expects two string values. It will split the string by whitespace
        and compare each value. It will return True if both lists are the same,
        contain the same elements and the same order."""
        if a is None or b is None:
            return False
        a = a.split()
        b = b.split()
        if len(a) != len(b):
            return False
        return len([i for i, j in zip(a, b) if i == j]) == len(a)

    def _parse_value(self, value):
        if value is None:
            return ''
        elif isinstance(value, bool):
            if value:
                return '1'
            else:
                return '0'
        elif isinstance(value, string_types):
            if value.lower() in BOOLEANS_TRUE:
                return '1'
            elif value.lower() in BOOLEANS_FALSE:
                return '0'
            else:
                return value.strip()
        else:
            return value

    def _stderr_failed(self, err):
        errors_regex = '^sysctl: setting key "[^"]+": (Invalid argument|Read-only file system)$'
        return re.search(errors_regex, err, re.MULTILINE) is not None

    def get_token_curr_value(self, token):
        if self.platform == 'openbsd':
            thiscmd = '%s -n %s' % (self.sysctl_cmd, token)
        else:
            thiscmd = '%s -e -n %s' % (self.sysctl_cmd, token)
        rc, out, err = self.module.run_command(thiscmd, environ_update=self.LANG_ENV)
        if rc != 0:
            return None
        else:
            return out

    def set_token_value(self, token, value):
        if len(value.split()) > 0:
            value = '"' + value + '"'
        if self.platform == 'openbsd':
            thiscmd = '%s %s=%s' % (self.sysctl_cmd, token, value)
        elif self.platform == 'freebsd':
            ignore_missing = ''
            if self.args['ignoreerrors']:
                ignore_missing = '-i'
            thiscmd = '%s %s %s=%s' % (self.sysctl_cmd, ignore_missing, token, value)
        else:
            ignore_missing = ''
            if self.args['ignoreerrors']:
                ignore_missing = '-e'
            thiscmd = '%s %s -w %s=%s' % (self.sysctl_cmd, ignore_missing, token, value)
        rc, out, err = self.module.run_command(thiscmd, environ_update=self.LANG_ENV)
        if rc != 0 or self._stderr_failed(err):
            self.module.fail_json(msg='setting %s failed: %s' % (token, out + err))
        else:
            return rc

    def reload_sysctl(self):
        if self.platform == 'freebsd':
            rc, out, err = self.module.run_command('/etc/rc.d/sysctl reload', environ_update=self.LANG_ENV)
        elif self.platform == 'openbsd':
            for k, v in self.file_values.items():
                rc = 0
                if k != self.args['name']:
                    rc = self.set_token_value(k, v)
                    if rc != 0:
                        break
            if rc == 0 and self.args['state'] == 'present':
                rc = self.set_token_value(self.args['name'], self.args['value'])
            return
        else:
            sysctl_args = [self.sysctl_cmd, '-p', self.sysctl_file]
            if self.args['ignoreerrors']:
                sysctl_args.insert(1, '-e')
            rc, out, err = self.module.run_command(sysctl_args, environ_update=self.LANG_ENV)
        if rc != 0 or self._stderr_failed(err):
            self.module.fail_json(msg='Failed to reload sysctl: %s' % to_native(out) + to_native(err))

    def read_sysctl_file(self):
        lines = []
        if os.path.isfile(self.sysctl_file):
            try:
                with open(self.sysctl_file, 'r') as read_file:
                    lines = read_file.readlines()
            except IOError as e:
                self.module.fail_json(msg='Failed to open %s: %s' % (to_native(self.sysctl_file), to_native(e)))
        for line in lines:
            line = line.strip()
            self.file_lines.append(line)
            if not line or line.startswith(('#', ';')) or '=' not in line:
                continue
            k, v = line.split('=', 1)
            k = k.strip()
            v = v.strip()
            self.file_values[k] = v.strip()

    def fix_lines(self):
        checked = []
        self.fixed_lines = []
        for line in self.file_lines:
            if not line.strip() or line.strip().startswith(('#', ';')) or '=' not in line:
                self.fixed_lines.append(line)
                continue
            tmpline = line.strip()
            k, v = tmpline.split('=', 1)
            k = k.strip()
            v = v.strip()
            if k not in checked:
                checked.append(k)
                if k == self.args['name']:
                    if self.args['state'] == 'present':
                        new_line = '%s=%s\n' % (k, self.args['value'])
                        self.fixed_lines.append(new_line)
                else:
                    new_line = '%s=%s\n' % (k, v)
                    self.fixed_lines.append(new_line)
        if self.args['name'] not in checked and self.args['state'] == 'present':
            new_line = '%s=%s\n' % (self.args['name'], self.args['value'])
            self.fixed_lines.append(new_line)

    def write_sysctl(self):
        fd, tmp_path = tempfile.mkstemp('.conf', '.ansible_m_sysctl_', os.path.dirname(self.sysctl_file))
        f = open(tmp_path, 'w')
        try:
            for l in self.fixed_lines:
                f.write(l.strip() + '\n')
        except IOError as e:
            self.module.fail_json(msg='Failed to write to file %s: %s' % (tmp_path, to_native(e)))
        f.flush()
        f.close()
        self.module.atomic_move(tmp_path, self.sysctl_file)