from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import re
class Sysrc(object):

    def __init__(self, module, name, value, path, delim, jail):
        self.module = module
        self.name = name
        self.changed = False
        self.value = value
        self.path = path
        self.delim = delim
        self.jail = jail
        self.sysrc = module.get_bin_path('sysrc', True)

    def has_unknown_variable(self, out, err):
        return err.find('unknown variable') > 0 or out.find('unknown variable') > 0

    def exists(self):
        rc, out, err = self.run_sysrc(self.name)
        if self.value is None:
            regex = '%s: ' % re.escape(self.name)
        else:
            regex = '%s: %s$' % (re.escape(self.name), re.escape(self.value))
        return not self.has_unknown_variable(out, err) and re.match(regex, out) is not None

    def contains(self):
        rc, out, err = self.run_sysrc('-n', self.name)
        if self.has_unknown_variable(out, err):
            return False
        return self.value in out.strip().split(self.delim)

    def present(self):
        if self.exists():
            return
        if self.module.check_mode:
            self.changed = True
            return
        rc, out, err = self.run_sysrc('%s=%s' % (self.name, self.value))
        if out.find('%s:' % self.name) == 0 and re.search('-> %s$' % re.escape(self.value), out) is not None:
            self.changed = True

    def absent(self):
        if not self.exists():
            return
        if not self.module.check_mode:
            rc, out, err = self.run_sysrc('-x', self.name)
            if self.has_unknown_variable(out, err):
                return
        self.changed = True

    def value_present(self):
        if self.contains():
            return
        if self.module.check_mode:
            self.changed = True
            return
        setstring = '%s+=%s%s' % (self.name, self.delim, self.value)
        rc, out, err = self.run_sysrc(setstring)
        if out.find('%s:' % self.name) == 0:
            values = out.split(' -> ')[1].strip().split(self.delim)
            if self.value in values:
                self.changed = True

    def value_absent(self):
        if not self.contains():
            return
        if self.module.check_mode:
            self.changed = True
            return
        setstring = '%s-=%s%s' % (self.name, self.delim, self.value)
        rc, out, err = self.run_sysrc(setstring)
        if out.find('%s:' % self.name) == 0:
            values = out.split(' -> ')[1].strip().split(self.delim)
            if self.value not in values:
                self.changed = True

    def run_sysrc(self, *args):
        cmd = [self.sysrc, '-f', self.path]
        if self.jail:
            cmd += ['-j', self.jail]
        cmd.extend(args)
        rc, out, err = self.module.run_command(cmd)
        return (rc, out, err)