from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import os
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
import platform
class Mas(object):

    def __init__(self, module):
        self.module = module
        self.mas_path = self.module.get_bin_path('mas')
        self._checked_signin = False
        self._mac_version = platform.mac_ver()[0] or '0.0'
        self._installed = None
        self._outdated = None
        self.count_install = 0
        self.count_upgrade = 0
        self.count_uninstall = 0
        self.result = {'changed': False}
        self.check_mas_tool()

    def app_command(self, command, id):
        """ Runs a `mas` command on a given app; command can be 'install', 'upgrade' or 'uninstall' """
        if not self.module.check_mode:
            if command != 'uninstall':
                self.check_signin()
            rc, out, err = self.run([command, str(id)])
            if rc != 0:
                self.module.fail_json(msg="Error running command '{0}' on app '{1}': {2}".format(command, str(id), out.rstrip()))
        self.__dict__['count_' + command] += 1

    def check_mas_tool(self):
        """ Verifies that the `mas` tool is available in a recent version """
        if not self.mas_path:
            self.module.fail_json(msg='Required `mas` tool is not installed')
        rc, out, err = self.run(['version'])
        if rc != 0 or not out.strip() or LooseVersion(out.strip()) < LooseVersion('1.5.0'):
            self.module.fail_json(msg='`mas` tool in version 1.5.0+ needed, got ' + out.strip())

    def check_signin(self):
        """ Verifies that the user is signed in to the Mac App Store """
        if self._checked_signin:
            return
        if LooseVersion(self._mac_version) >= LooseVersion(NOT_WORKING_MAC_VERSION_MAS_ACCOUNT):
            self.module.log('WARNING: You must be signed in via the Mac App Store GUI beforehand else error will occur')
        else:
            rc, out, err = self.run(['account'])
            if out.split('\n', 1)[0].rstrip() == 'Not signed in':
                self.module.fail_json(msg='You must be signed in to the Mac App Store')
        self._checked_signin = True

    def exit(self):
        """ Exit with the data we have collected over time """
        msgs = []
        if self.count_install > 0:
            msgs.append('Installed {0} app(s)'.format(self.count_install))
        if self.count_upgrade > 0:
            msgs.append('Upgraded {0} app(s)'.format(self.count_upgrade))
        if self.count_uninstall > 0:
            msgs.append('Uninstalled {0} app(s)'.format(self.count_uninstall))
        if msgs:
            self.result['changed'] = True
            self.result['msg'] = ', '.join(msgs)
        self.module.exit_json(**self.result)

    def get_current_state(self, command):
        """ Returns the list of all app IDs; command can either be 'list' or 'outdated' """
        rc, raw_apps, err = self.run([command])
        rows = raw_apps.split('\n')
        if rows[0] == 'No installed apps found':
            rows = []
        apps = []
        for r in rows:
            r = r.split(' ', 1)
            if len(r) == 2:
                apps.append(int(r[0]))
        return apps

    def installed(self):
        """ Returns the list of installed apps """
        if self._installed is None:
            self._installed = self.get_current_state('list')
        return self._installed

    def is_installed(self, id):
        """ Checks whether the given app is installed """
        return int(id) in self.installed()

    def is_outdated(self, id):
        """ Checks whether the given app is installed, but outdated """
        return int(id) in self.outdated()

    def outdated(self):
        """ Returns the list of installed, but outdated apps """
        if self._outdated is None:
            self._outdated = self.get_current_state('outdated')
        return self._outdated

    def run(self, cmd):
        """ Runs a command of the `mas` tool """
        cmd.insert(0, self.mas_path)
        return self.module.run_command(cmd, False)

    def upgrade_all(self):
        """ Upgrades all installed apps and sets the correct result data """
        outdated = self.outdated()
        if not self.module.check_mode:
            self.check_signin()
            rc, out, err = self.run(['upgrade'])
            if rc != 0:
                self.module.fail_json(msg='Could not upgrade all apps: ' + out.rstrip())
        self.count_upgrade += len(outdated)