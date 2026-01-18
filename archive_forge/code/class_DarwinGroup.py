from __future__ import absolute_import, division, print_function
import grp
import os
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.sys_info import get_platform_subclass
class DarwinGroup(Group):
    """
    This is a Mac macOS Darwin Group manipulation class.

    This overrides the following methods from the generic class:-
      - group_del()
      - group_add()
      - group_mod()

    group manipulation are done using dseditgroup(1).
    """
    platform = 'Darwin'
    distribution = None

    def group_add(self, **kwargs):
        cmd = [self.module.get_bin_path('dseditgroup', True)]
        cmd += ['-o', 'create']
        if self.gid is not None:
            cmd += ['-i', str(self.gid)]
        elif 'system' in kwargs and kwargs['system'] is True:
            gid = self.get_lowest_available_system_gid()
            if gid is not False:
                self.gid = str(gid)
                cmd += ['-i', str(self.gid)]
        cmd += ['-L', self.name]
        rc, out, err = self.execute_command(cmd)
        return (rc, out, err)

    def group_del(self):
        cmd = [self.module.get_bin_path('dseditgroup', True)]
        cmd += ['-o', 'delete']
        cmd += ['-L', self.name]
        rc, out, err = self.execute_command(cmd)
        return (rc, out, err)

    def group_mod(self, gid=None):
        info = self.group_info()
        if self.gid is not None and int(self.gid) != info[2]:
            cmd = [self.module.get_bin_path('dseditgroup', True)]
            cmd += ['-o', 'edit']
            if gid is not None:
                cmd += ['-i', str(gid)]
            cmd += ['-L', self.name]
            rc, out, err = self.execute_command(cmd)
            return (rc, out, err)
        return (None, '', '')

    def get_lowest_available_system_gid(self):
        try:
            cmd = [self.module.get_bin_path('dscl', True)]
            cmd += ['/Local/Default', '-list', '/Groups', 'PrimaryGroupID']
            rc, out, err = self.execute_command(cmd)
            lines = out.splitlines()
            highest = 0
            for group_info in lines:
                parts = group_info.split(' ')
                if len(parts) > 1:
                    gid = int(parts[-1])
                    if gid > highest and gid < 500:
                        highest = gid
            if highest == 0 or highest == 499:
                return False
            return highest + 1
        except Exception:
            return False