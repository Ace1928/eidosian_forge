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
class FreeBsdUser(User):
    """
    This is a FreeBSD User manipulation class - it uses the pw command
    to manipulate the user database, followed by the chpass command
    to change the password.

    This overrides the following methods from the generic class:-
      - create_user()
      - remove_user()
      - modify_user()
    """
    platform = 'FreeBSD'
    distribution = None
    SHADOWFILE = '/etc/master.passwd'
    SHADOWFILE_EXPIRE_INDEX = 6
    DATE_FORMAT = '%d-%b-%Y'

    def _handle_lock(self):
        info = self.user_info()
        if self.password_lock and (not info[1].startswith('*LOCKED*')):
            cmd = [self.module.get_bin_path('pw', True), 'lock', self.name]
            if self.uid is not None and info[2] != int(self.uid):
                cmd.append('-u')
                cmd.append(self.uid)
            return self.execute_command(cmd)
        elif self.password_lock is False and info[1].startswith('*LOCKED*'):
            cmd = [self.module.get_bin_path('pw', True), 'unlock', self.name]
            if self.uid is not None and info[2] != int(self.uid):
                cmd.append('-u')
                cmd.append(self.uid)
            return self.execute_command(cmd)
        return (None, '', '')

    def remove_user(self):
        cmd = [self.module.get_bin_path('pw', True), 'userdel', '-n', self.name]
        if self.remove:
            cmd.append('-r')
        return self.execute_command(cmd)

    def create_user(self):
        cmd = [self.module.get_bin_path('pw', True), 'useradd', '-n', self.name]
        if self.uid is not None:
            cmd.append('-u')
            cmd.append(self.uid)
            if self.non_unique:
                cmd.append('-o')
        if self.comment is not None:
            cmd.append('-c')
            cmd.append(self.comment)
        if self.home is not None:
            cmd.append('-d')
            cmd.append(self.home)
        if self.group is not None:
            if not self.group_exists(self.group):
                self.module.fail_json(msg='Group %s does not exist' % self.group)
            cmd.append('-g')
            cmd.append(self.group)
        if self.groups is not None:
            groups = self.get_groups_set()
            cmd.append('-G')
            cmd.append(','.join(groups))
        if self.create_home:
            cmd.append('-m')
            if self.skeleton is not None:
                cmd.append('-k')
                cmd.append(self.skeleton)
            if self.umask is not None:
                cmd.append('-K')
                cmd.append('UMASK=' + self.umask)
        if self.shell is not None:
            cmd.append('-s')
            cmd.append(self.shell)
        if self.login_class is not None:
            cmd.append('-L')
            cmd.append(self.login_class)
        if self.expires is not None:
            cmd.append('-e')
            if self.expires < time.gmtime(0):
                cmd.append('0')
            else:
                cmd.append(str(calendar.timegm(self.expires)))
        rc, out, err = self.execute_command(cmd)
        if rc is not None and rc != 0:
            self.module.fail_json(name=self.name, msg=err, rc=rc)
        if self.password is not None:
            cmd = [self.module.get_bin_path('chpass', True), '-p', self.password, self.name]
            _rc, _out, _err = self.execute_command(cmd)
            if rc is None:
                rc = _rc
            out += _out
            err += _err
        _rc, _out, _err = self._handle_lock()
        if rc is None:
            rc = _rc
        out += _out
        err += _err
        return (rc, out, err)

    def modify_user(self):
        cmd = [self.module.get_bin_path('pw', True), 'usermod', '-n', self.name]
        cmd_len = len(cmd)
        info = self.user_info()
        if self.uid is not None and info[2] != int(self.uid):
            cmd.append('-u')
            cmd.append(self.uid)
            if self.non_unique:
                cmd.append('-o')
        if self.comment is not None and info[4] != self.comment:
            cmd.append('-c')
            cmd.append(self.comment)
        if self.home is not None:
            if info[5] != self.home and self.move_home or (not os.path.exists(self.home) and self.create_home):
                cmd.append('-m')
            if info[5] != self.home:
                cmd.append('-d')
                cmd.append(self.home)
            if self.skeleton is not None:
                cmd.append('-k')
                cmd.append(self.skeleton)
            if self.umask is not None:
                cmd.append('-K')
                cmd.append('UMASK=' + self.umask)
        if self.group is not None:
            if not self.group_exists(self.group):
                self.module.fail_json(msg='Group %s does not exist' % self.group)
            ginfo = self.group_info(self.group)
            if info[3] != ginfo[2]:
                cmd.append('-g')
                cmd.append(self.group)
        if self.shell is not None and info[6] != self.shell:
            cmd.append('-s')
            cmd.append(self.shell)
        if self.login_class is not None:
            user_login_class = None
            if os.path.exists(self.SHADOWFILE) and os.access(self.SHADOWFILE, os.R_OK):
                with open(self.SHADOWFILE, 'r') as f:
                    for line in f:
                        if line.startswith('%s:' % self.name):
                            user_login_class = line.split(':')[4]
            if self.login_class != user_login_class:
                cmd.append('-L')
                cmd.append(self.login_class)
        if self.groups is not None:
            current_groups = self.user_group_membership()
            groups = self.get_groups_set(names_only=True)
            group_diff = set(current_groups).symmetric_difference(groups)
            groups_need_mod = False
            if group_diff:
                if self.append:
                    for g in groups:
                        if g in group_diff:
                            groups_need_mod = True
                            break
                else:
                    groups_need_mod = True
            if groups_need_mod:
                cmd.append('-G')
                new_groups = groups
                if self.append:
                    new_groups = groups | set(current_groups)
                cmd.append(','.join(new_groups))
        if self.expires is not None:
            current_expires = self.user_password()[1] or '0'
            current_expires = int(current_expires)
            if self.expires <= time.gmtime(0):
                if current_expires > 0:
                    cmd.append('-e')
                    cmd.append('0')
            else:
                current_expire_date = time.gmtime(current_expires)
                if current_expires <= 0 or current_expire_date[:3] != self.expires[:3]:
                    cmd.append('-e')
                    cmd.append(str(calendar.timegm(self.expires)))
        rc, out, err = (None, '', '')
        if cmd_len != len(cmd):
            rc, _out, _err = self.execute_command(cmd)
            out += _out
            err += _err
            if rc is not None and rc != 0:
                self.module.fail_json(name=self.name, msg=err, rc=rc)
        if self.update_password == 'always' and self.password is not None and (info[1].lstrip('*LOCKED*') != self.password.lstrip('*LOCKED*')):
            cmd = [self.module.get_bin_path('chpass', True), '-p', self.password, self.name]
            _rc, _out, _err = self.execute_command(cmd)
            if rc is None:
                rc = _rc
            out += _out
            err += _err
        _rc, _out, _err = self._handle_lock()
        if rc is None:
            rc = _rc
        out += _out
        err += _err
        return (rc, out, err)