from __future__ import absolute_import, division, print_function
import ctypes.util
import grp
import calendar
import os
import re
import pty
import pwd
import select
import shutil
import socket
import subprocess
import time
import math
from ansible.module_utils import distro
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.common.sys_info import get_platform_subclass
import ansible.module_utils.compat.typing as t
class DarwinUser(User):
    """
    This is a Darwin macOS User manipulation class.
    Main differences are that Darwin:-
      - Handles accounts in a database managed by dscl(1)
      - Has no useradd/groupadd
      - Does not create home directories
      - User password must be cleartext
      - UID must be given
      - System users must ben under 500

    This overrides the following methods from the generic class:-
      - user_exists()
      - create_user()
      - remove_user()
      - modify_user()
    """
    platform = 'Darwin'
    distribution = None
    SHADOWFILE = None
    dscl_directory = '.'
    fields = [('comment', 'RealName'), ('home', 'NFSHomeDirectory'), ('shell', 'UserShell'), ('uid', 'UniqueID'), ('group', 'PrimaryGroupID'), ('hidden', 'IsHidden')]

    def __init__(self, module):
        super(DarwinUser, self).__init__(module)
        if self.hidden is None:
            if self.system:
                self.hidden = 1
        elif self.hidden:
            self.hidden = 1
        else:
            self.hidden = 0
        if self.hidden is not None:
            self.fields.append(('hidden', 'IsHidden'))

    def _get_dscl(self):
        return [self.module.get_bin_path('dscl', True), self.dscl_directory]

    def _list_user_groups(self):
        cmd = self._get_dscl()
        cmd += ['-search', '/Groups', 'GroupMembership', self.name]
        rc, out, err = self.execute_command(cmd, obey_checkmode=False)
        groups = []
        for line in out.splitlines():
            if line.startswith(' ') or line.startswith(')'):
                continue
            groups.append(line.split()[0])
        return groups

    def _get_user_property(self, property):
        """Return user PROPERTY as given my dscl(1) read or None if not found."""
        cmd = self._get_dscl()
        cmd += ['-read', '/Users/%s' % self.name, property]
        rc, out, err = self.execute_command(cmd, obey_checkmode=False)
        if rc != 0:
            return None
        lines = out.splitlines()
        if len(lines) == 1:
            return lines[0].split(': ')[1]
        if len(lines) > 2:
            return '\n'.join([lines[1].strip()] + lines[2:])
        if len(lines) == 2:
            return lines[1].strip()
        return None

    def _get_next_uid(self, system=None):
        """
        Return the next available uid. If system=True, then
        uid should be below of 500, if possible.
        """
        cmd = self._get_dscl()
        cmd += ['-list', '/Users', 'UniqueID']
        rc, out, err = self.execute_command(cmd, obey_checkmode=False)
        if rc != 0:
            self.module.fail_json(msg='Unable to get the next available uid', rc=rc, out=out, err=err)
        max_uid = 0
        max_system_uid = 0
        for line in out.splitlines():
            current_uid = int(line.split(' ')[-1])
            if max_uid < current_uid:
                max_uid = current_uid
            if max_system_uid < current_uid and current_uid < 500:
                max_system_uid = current_uid
        if system and 0 < max_system_uid < 499:
            return max_system_uid + 1
        return max_uid + 1

    def _change_user_password(self):
        """Change password for SELF.NAME against SELF.PASSWORD.

        Please note that password must be cleartext.
        """
        cmd = self._get_dscl()
        if self.password:
            cmd += ['-passwd', '/Users/%s' % self.name, self.password]
        else:
            cmd += ['-create', '/Users/%s' % self.name, 'Password', '*']
        rc, out, err = self.execute_command(cmd)
        if rc != 0:
            self.module.fail_json(msg='Error when changing password', err=err, out=out, rc=rc)
        return (rc, out, err)

    def _make_group_numerical(self):
        """Convert SELF.GROUP to is stringed numerical value suitable for dscl."""
        if self.group is None:
            self.group = 'nogroup'
        try:
            self.group = grp.getgrnam(self.group).gr_gid
        except KeyError:
            self.module.fail_json(msg='Group "%s" not found. Try to create it first using "group" module.' % self.group)
        self.group = str(self.group)

    def __modify_group(self, group, action):
        """Add or remove SELF.NAME to or from GROUP depending on ACTION.
        ACTION can be 'add' or 'remove' otherwise 'remove' is assumed. """
        if action == 'add':
            option = '-a'
        else:
            option = '-d'
        cmd = ['dseditgroup', '-o', 'edit', option, self.name, '-t', 'user', group]
        rc, out, err = self.execute_command(cmd)
        if rc != 0:
            self.module.fail_json(msg='Cannot %s user "%s" to group "%s".' % (action, self.name, group), err=err, out=out, rc=rc)
        return (rc, out, err)

    def _modify_group(self):
        """Add or remove SELF.NAME to or from GROUP depending on ACTION.
        ACTION can be 'add' or 'remove' otherwise 'remove' is assumed. """
        rc = 0
        out = ''
        err = ''
        changed = False
        current = set(self._list_user_groups())
        if self.groups is not None:
            target = self.get_groups_set(names_only=True)
        else:
            target = set([])
        if self.append is False:
            for remove in current - target:
                _rc, _out, _err = self.__modify_group(remove, 'delete')
                rc += rc
                out += _out
                err += _err
                changed = True
        for add in target - current:
            _rc, _out, _err = self.__modify_group(add, 'add')
            rc += _rc
            out += _out
            err += _err
            changed = True
        return (rc, out, err, changed)

    def _update_system_user(self):
        """Hide or show user on login window according SELF.SYSTEM.

        Returns 0 if a change has been made, None otherwise."""
        plist_file = '/Library/Preferences/com.apple.loginwindow.plist'
        cmd = ['defaults', 'read', plist_file, 'HiddenUsersList']
        rc, out, err = self.execute_command(cmd, obey_checkmode=False)
        hidden_users = []
        for x in out.splitlines()[1:-1]:
            try:
                x = x.split('"')[1]
            except IndexError:
                x = x.strip()
            hidden_users.append(x)
        if self.system:
            if self.name not in hidden_users:
                cmd = ['defaults', 'write', plist_file, 'HiddenUsersList', '-array-add', self.name]
                rc, out, err = self.execute_command(cmd)
                if rc != 0:
                    self.module.fail_json(msg='Cannot user "%s" to hidden user list.' % self.name, err=err, out=out, rc=rc)
                return 0
        elif self.name in hidden_users:
            del hidden_users[hidden_users.index(self.name)]
            cmd = ['defaults', 'write', plist_file, 'HiddenUsersList', '-array'] + hidden_users
            rc, out, err = self.execute_command(cmd)
            if rc != 0:
                self.module.fail_json(msg='Cannot remove user "%s" from hidden user list.' % self.name, err=err, out=out, rc=rc)
            return 0

    def user_exists(self):
        """Check is SELF.NAME is a known user on the system."""
        cmd = self._get_dscl()
        cmd += ['-read', '/Users/%s' % self.name, 'UniqueID']
        rc, out, err = self.execute_command(cmd, obey_checkmode=False)
        return rc == 0

    def remove_user(self):
        """Delete SELF.NAME. If SELF.FORCE is true, remove its home directory."""
        info = self.user_info()
        cmd = self._get_dscl()
        cmd += ['-delete', '/Users/%s' % self.name]
        rc, out, err = self.execute_command(cmd)
        if rc != 0:
            self.module.fail_json(msg='Cannot delete user "%s".' % self.name, err=err, out=out, rc=rc)
        if self.force:
            if os.path.exists(info[5]):
                shutil.rmtree(info[5])
                out += 'Removed %s' % info[5]
        return (rc, out, err)

    def create_user(self, command_name='dscl'):
        cmd = self._get_dscl()
        cmd += ['-create', '/Users/%s' % self.name]
        rc, out, err = self.execute_command(cmd)
        if rc != 0:
            self.module.fail_json(msg='Cannot create user "%s".' % self.name, err=err, out=out, rc=rc)
        if self.comment is None:
            self.comment = self.name
        if self.group is None:
            self.group = 'staff'
        self._make_group_numerical()
        if self.uid is None:
            self.uid = str(self._get_next_uid(self.system))
        if self.create_home:
            if self.home is None:
                self.home = '/Users/%s' % self.name
            if not self.module.check_mode:
                if not os.path.exists(self.home):
                    os.makedirs(self.home)
                self.chown_homedir(int(self.uid), int(self.group), self.home)
        if not self.system and self.shell is None:
            self.shell = '/bin/bash'
        for field in self.fields:
            if field[0] in self.__dict__ and self.__dict__[field[0]]:
                cmd = self._get_dscl()
                cmd += ['-create', '/Users/%s' % self.name, field[1], self.__dict__[field[0]]]
                rc, _out, _err = self.execute_command(cmd)
                if rc != 0:
                    self.module.fail_json(msg='Cannot add property "%s" to user "%s".' % (field[0], self.name), err=err, out=out, rc=rc)
                out += _out
                err += _err
                if rc != 0:
                    return (rc, _out, _err)
        rc, _out, _err = self._change_user_password()
        out += _out
        err += _err
        self._update_system_user()
        if self.groups:
            rc, _out, _err, changed = self._modify_group()
            out += _out
            err += _err
        return (rc, out, err)

    def modify_user(self):
        changed = None
        out = ''
        err = ''
        if self.group:
            self._make_group_numerical()
        for field in self.fields:
            if field[0] in self.__dict__ and self.__dict__[field[0]]:
                current = self._get_user_property(field[1])
                if current is None or current != to_text(self.__dict__[field[0]]):
                    cmd = self._get_dscl()
                    cmd += ['-create', '/Users/%s' % self.name, field[1], self.__dict__[field[0]]]
                    rc, _out, _err = self.execute_command(cmd)
                    if rc != 0:
                        self.module.fail_json(msg='Cannot update property "%s" for user "%s".' % (field[0], self.name), err=err, out=out, rc=rc)
                    changed = rc
                    out += _out
                    err += _err
        if self.update_password == 'always' and self.password is not None:
            rc, _out, _err = self._change_user_password()
            out += _out
            err += _err
            changed = rc
        if self.groups:
            rc, _out, _err, _changed = self._modify_group()
            out += _out
            err += _err
            if _changed is True:
                changed = rc
        rc = self._update_system_user()
        if rc == 0:
            changed = rc
        return (changed, out, err)