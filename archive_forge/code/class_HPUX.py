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
class HPUX(User):
    """
    This is a HP-UX User manipulation class.

    This overrides the following methods from the generic class:-
      - create_user()
      - remove_user()
      - modify_user()
    """
    platform = 'HP-UX'
    distribution = None
    SHADOWFILE = '/etc/shadow'

    def create_user(self):
        cmd = ['/usr/sam/lbin/useradd.sam']
        if self.uid is not None:
            cmd.append('-u')
            cmd.append(self.uid)
            if self.non_unique:
                cmd.append('-o')
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
        if self.password is not None:
            cmd.append('-p')
            cmd.append(self.password)
        if self.create_home:
            cmd.append('-m')
        else:
            cmd.append('-M')
        if self.system:
            cmd.append('-r')
        cmd.append(self.name)
        return self.execute_command(cmd)

    def remove_user(self):
        cmd = ['/usr/sam/lbin/userdel.sam']
        if self.force:
            cmd.append('-F')
        if self.remove:
            cmd.append('-r')
        cmd.append(self.name)
        return self.execute_command(cmd)

    def modify_user(self):
        cmd = ['/usr/sam/lbin/usermod.sam']
        info = self.user_info()
        if self.uid is not None and info[2] != int(self.uid):
            cmd.append('-u')
            cmd.append(self.uid)
            if self.non_unique:
                cmd.append('-o')
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
                groups = self.get_groups_set(remove_existing=False, names_only=True)
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
                new_groups = groups
                if self.append:
                    new_groups = groups | set(current_groups)
                cmd.append(','.join(new_groups))
        if self.comment is not None and info[4] != self.comment:
            cmd.append('-c')
            cmd.append(self.comment)
        if self.home is not None and info[5] != self.home:
            cmd.append('-d')
            cmd.append(self.home)
            if self.move_home:
                cmd.append('-m')
        if self.shell is not None and info[6] != self.shell:
            cmd.append('-s')
            cmd.append(self.shell)
        if self.update_password == 'always' and self.password is not None and (info[1] != self.password):
            cmd.append('-F')
            cmd.append('-p')
            cmd.append(self.password)
        if len(cmd) == 1:
            return (None, '', '')
        cmd.append(self.name)
        return self.execute_command(cmd)