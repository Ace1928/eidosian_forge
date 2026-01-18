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
class AIX(User):
    """
    This is a AIX User manipulation class.

    This overrides the following methods from the generic class:-
      - create_user()
      - remove_user()
      - modify_user()
      - parse_shadow_file()
    """
    platform = 'AIX'
    distribution = None
    SHADOWFILE = '/etc/security/passwd'

    def remove_user(self):
        cmd = [self.module.get_bin_path('userdel', True)]
        if self.remove:
            cmd.append('-r')
        cmd.append(self.name)
        return self.execute_command(cmd)

    def create_user_useradd(self, command_name='useradd'):
        cmd = [self.module.get_bin_path(command_name, True)]
        if self.uid is not None:
            cmd.append('-u')
            cmd.append(self.uid)
        if self.group is not None:
            if not self.group_exists(self.group):
                self.module.fail_json(msg='Group %s does not exist' % self.group)
            cmd.append('-g')
            cmd.append(self.group)
        if self.groups is not None and len(self.groups):
            groups = self.get_groups_set()
            cmd.append('-G')
            cmd.append(','.join(groups))
        if self.comment is not None:
            cmd.append('-c')
            cmd.append(self.comment)
        if self.home is not None:
            cmd.append('-d')
            cmd.append(self.home)
        if self.shell is not None:
            cmd.append('-s')
            cmd.append(self.shell)
        if self.create_home:
            cmd.append('-m')
            if self.skeleton is not None:
                cmd.append('-k')
                cmd.append(self.skeleton)
            if self.umask is not None:
                cmd.append('-K')
                cmd.append('UMASK=' + self.umask)
        cmd.append(self.name)
        rc, out, err = self.execute_command(cmd)
        if self.password is not None:
            cmd = []
            cmd.append(self.module.get_bin_path('chpasswd', True))
            cmd.append('-e')
            cmd.append('-c')
            self.execute_command(cmd, data='%s:%s' % (self.name, self.password))
        return (rc, out, err)

    def modify_user_usermod(self):
        cmd = [self.module.get_bin_path('usermod', True)]
        info = self.user_info()
        if self.uid is not None and info[2] != int(self.uid):
            cmd.append('-u')
            cmd.append(self.uid)
        if self.group is not None:
            if not self.group_exists(self.group):
                self.module.fail_json(msg='Group %s does not exist' % self.group)
            ginfo = self.group_info(self.group)
            if info[3] != ginfo[2]:
                cmd.append('-g')
                cmd.append(self.group)
        if self.groups is not None:
            current_groups = self.user_group_membership()
            groups_need_mod = False
            groups = []
            if self.groups == '':
                if current_groups and (not self.append):
                    groups_need_mod = True
            else:
                groups = self.get_groups_set(names_only=True)
                group_diff = set(current_groups).symmetric_difference(groups)
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
                cmd.append(','.join(groups))
        if self.comment is not None and info[4] != self.comment:
            cmd.append('-c')
            cmd.append(self.comment)
        if self.home is not None and info[5] != self.home:
            if self.move_home:
                cmd.append('-m')
            cmd.append('-d')
            cmd.append(self.home)
        if self.shell is not None and info[6] != self.shell:
            cmd.append('-s')
            cmd.append(self.shell)
        if len(cmd) == 1:
            rc, out, err = (None, '', '')
        else:
            cmd.append(self.name)
            rc, out, err = self.execute_command(cmd)
        if self.update_password == 'always' and self.password is not None and (info[1] != self.password):
            cmd = []
            cmd.append(self.module.get_bin_path('chpasswd', True))
            cmd.append('-e')
            cmd.append('-c')
            rc2, out2, err2 = self.execute_command(cmd, data='%s:%s' % (self.name, self.password))
        else:
            rc2, out2, err2 = (None, '', '')
        if rc is not None:
            return (rc, out + out2, err + err2)
        else:
            return (rc2, out + out2, err + err2)

    def parse_shadow_file(self):
        """Example AIX shadowfile data:
        nobody:
                password = *

        operator1:
                password = {ssha512}06$xxxxxxxxxxxx....
                lastupdate = 1549558094

        test1:
                password = *
                lastupdate = 1553695126

        """
        b_name = to_bytes(self.name)
        b_passwd = b''
        b_expires = b''
        if os.path.exists(self.SHADOWFILE) and os.access(self.SHADOWFILE, os.R_OK):
            with open(self.SHADOWFILE, 'rb') as bf:
                b_lines = bf.readlines()
            b_passwd_line = b''
            b_expires_line = b''
            try:
                for index, b_line in enumerate(b_lines):
                    if b_line.startswith(b'%s:' % b_name):
                        b_passwd_line = b_lines[index + 1]
                        b_expires_line = b_lines[index + 2]
                        break
                if b' = ' in b_passwd_line:
                    b_passwd = b_passwd_line.split(b' = ', 1)[-1].strip()
                if b' = ' in b_expires_line:
                    b_expires = b_expires_line.split(b' = ', 1)[-1].strip()
            except IndexError:
                self.module.fail_json(msg='Failed to parse shadow file %s' % self.SHADOWFILE)
        passwd = to_native(b_passwd)
        expires = to_native(b_expires) or -1
        return (passwd, expires)