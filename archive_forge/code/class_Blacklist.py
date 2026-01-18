from __future__ import absolute_import, division, print_function
import os
import re
from ansible_collections.community.general.plugins.module_utils.module_helper import StateModuleHelper
class Blacklist(StateModuleHelper):
    output_params = ('name', 'state')
    module = dict(argument_spec=dict(name=dict(type='str', required=True), state=dict(type='str', default='present', choices=['absent', 'present']), blacklist_file=dict(type='str', default='/etc/modprobe.d/blacklist-ansible.conf')), supports_check_mode=True)

    def __init_module__(self):
        self.pattern = re.compile('^blacklist\\s+{0}$'.format(re.escape(self.vars.name)))
        self.vars.filename = self.vars.blacklist_file
        self.vars.set('file_exists', os.path.exists(self.vars.filename), output=False, change=True)
        if not self.vars.file_exists:
            with open(self.vars.filename, 'a'):
                pass
            self.vars.file_exists = True
            self.vars.set('lines', [], change=True, diff=True)
        else:
            with open(self.vars.filename) as fd:
                self.vars.set('lines', [x.rstrip() for x in fd.readlines()], change=True, diff=True)
        self.vars.set('is_blacklisted', self._is_module_blocked(), change=True)

    def _is_module_blocked(self):
        for line in self.vars.lines:
            stripped = line.strip()
            if stripped.startswith('#'):
                continue
            if self.pattern.match(stripped):
                return True
        return False

    def state_absent(self):
        if not self.vars.is_blacklisted:
            return
        self.vars.is_blacklisted = False
        self.vars.lines = [line for line in self.vars.lines if not self.pattern.match(line.strip())]

    def state_present(self):
        if self.vars.is_blacklisted:
            return
        self.vars.is_blacklisted = True
        self.vars.lines = self.vars.lines + ['blacklist %s' % self.vars.name]

    def __quit_module__(self):
        if self.has_changed() and (not self.module.check_mode):
            bkp = self.module.backup_local(self.vars.filename)
            with open(self.vars.filename, 'w') as fd:
                fd.writelines(['{0}\n'.format(x) for x in self.vars.lines])
            self.module.add_cleanup_file(bkp)