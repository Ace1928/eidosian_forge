from __future__ import absolute_import, division, print_function
import grp
import os
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.sys_info import get_platform_subclass
class OpenBsdGroup(Group):
    """
    This is a OpenBSD Group manipulation class.

    This overrides the following methods from the generic class:-
      - group_del()
      - group_add()
      - group_mod()
    """
    platform = 'OpenBSD'
    distribution = None
    GROUPFILE = '/etc/group'

    def group_del(self):
        cmd = [self.module.get_bin_path('groupdel', True), self.name]
        return self.execute_command(cmd)

    def group_add(self, **kwargs):
        cmd = [self.module.get_bin_path('groupadd', True)]
        if self.gid is not None:
            cmd.append('-g')
            cmd.append(str(self.gid))
            if self.non_unique:
                cmd.append('-o')
        cmd.append(self.name)
        return self.execute_command(cmd)

    def group_mod(self, **kwargs):
        cmd = [self.module.get_bin_path('groupmod', True)]
        info = self.group_info()
        if self.gid is not None and int(self.gid) != info[2]:
            cmd.append('-g')
            cmd.append(str(self.gid))
            if self.non_unique:
                cmd.append('-o')
        if len(cmd) == 1:
            return (None, '', '')
        if self.module.check_mode:
            return (0, '', '')
        cmd.append(self.name)
        return self.execute_command(cmd)