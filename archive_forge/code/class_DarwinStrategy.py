from __future__ import absolute_import, division, print_function
import os
import platform
import socket
import traceback
import ansible.module_utils.compat.typing as t
from ansible.module_utils.basic import (
from ansible.module_utils.common.sys_info import get_platform_subclass
from ansible.module_utils.facts.system.service_mgr import ServiceMgrFactCollector
from ansible.module_utils.facts.utils import get_file_lines, get_file_content
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.six import PY3, text_type
class DarwinStrategy(BaseStrategy):
    """
    This is a macOS hostname manipulation strategy class. It uses
    /usr/sbin/scutil to set ComputerName, HostName, and LocalHostName.

    HostName corresponds to what most platforms consider to be hostname.
    It controls the name used on the command line and SSH.

    However, macOS also has LocalHostName and ComputerName settings.
    LocalHostName controls the Bonjour/ZeroConf name, used by services
    like AirDrop. This class implements a method, _scrub_hostname(), that mimics
    the transformations macOS makes on hostnames when enterened in the Sharing
    preference pane. It replaces spaces with dashes and removes all special
    characters.

    ComputerName is the name used for user-facing GUI services, like the
    System Preferences/Sharing pane and when users connect to the Mac over the network.
    """

    def __init__(self, module):
        super(DarwinStrategy, self).__init__(module)
        self.scutil = self.module.get_bin_path('scutil', True)
        self.name_types = ('HostName', 'ComputerName', 'LocalHostName')
        self.scrubbed_name = self._scrub_hostname(self.module.params['name'])

    def _make_translation(self, replace_chars, replacement_chars, delete_chars):
        if PY3:
            return str.maketrans(replace_chars, replacement_chars, delete_chars)
        if not isinstance(replace_chars, text_type) or not isinstance(replacement_chars, text_type):
            raise ValueError('replace_chars and replacement_chars must both be strings')
        if len(replace_chars) != len(replacement_chars):
            raise ValueError('replacement_chars must be the same length as replace_chars')
        table = dict(zip((ord(c) for c in replace_chars), replacement_chars))
        for char in delete_chars:
            table[ord(char)] = None
        return table

    def _scrub_hostname(self, name):
        """
        LocalHostName only accepts valid DNS characters while HostName and ComputerName
        accept a much wider range of characters. This function aims to mimic how macOS
        translates a friendly name to the LocalHostName.
        """
        name = to_text(name)
        replace_chars = u'\'"~`!@#$%^&*(){}[]/=?+\\|-_ '
        delete_chars = u".'"
        table = self._make_translation(replace_chars, u'-' * len(replace_chars), delete_chars)
        name = name.translate(table)
        while '-' * 2 in name:
            name = name.replace('-' * 2, '')
        name = name.rstrip('-')
        return name

    def get_current_hostname(self):
        cmd = [self.scutil, '--get', 'HostName']
        rc, out, err = self.module.run_command(cmd)
        if rc != 0 and 'HostName: not set' not in err:
            self.module.fail_json(msg='Failed to get current hostname rc=%d, out=%s, err=%s' % (rc, out, err))
        return to_native(out).strip()

    def get_permanent_hostname(self):
        cmd = [self.scutil, '--get', 'ComputerName']
        rc, out, err = self.module.run_command(cmd)
        if rc != 0:
            self.module.fail_json(msg='Failed to get permanent hostname rc=%d, out=%s, err=%s' % (rc, out, err))
        return to_native(out).strip()

    def set_permanent_hostname(self, name):
        for hostname_type in self.name_types:
            cmd = [self.scutil, '--set', hostname_type]
            if hostname_type == 'LocalHostName':
                cmd.append(to_native(self.scrubbed_name))
            else:
                cmd.append(to_native(name))
            rc, out, err = self.module.run_command(cmd)
            if rc != 0:
                self.module.fail_json(msg="Failed to set {3} to '{2}': {0} {1}".format(to_native(out), to_native(err), to_native(name), hostname_type))

    def set_current_hostname(self, name):
        pass

    def update_current_hostname(self):
        pass

    def update_permanent_hostname(self):
        name = self.module.params['name']
        all_names = tuple((self.module.run_command([self.scutil, '--get', name_type])[1].strip() for name_type in self.name_types))
        expected_names = tuple((self.scrubbed_name if n == 'LocalHostName' else name for n in self.name_types))
        if all_names != expected_names:
            if not self.module.check_mode:
                self.set_permanent_hostname(name)
            self.changed = True