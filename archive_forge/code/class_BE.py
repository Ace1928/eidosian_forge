from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
class BE(object):

    def __init__(self, module):
        self.module = module
        self.name = module.params['name']
        self.snapshot = module.params['snapshot']
        self.description = module.params['description']
        self.options = module.params['options']
        self.mountpoint = module.params['mountpoint']
        self.state = module.params['state']
        self.force = module.params['force']
        self.is_freebsd = os.uname()[0] == 'FreeBSD'

    def _beadm_list(self):
        cmd = [self.module.get_bin_path('beadm'), 'list', '-H']
        if '@' in self.name:
            cmd.append('-s')
        return self.module.run_command(cmd)

    def _find_be_by_name(self, out):
        if '@' in self.name:
            for line in out.splitlines():
                if self.is_freebsd:
                    check = line.split()
                    if check == []:
                        continue
                    full_name = check[0].split('/')
                    if full_name == []:
                        continue
                    check[0] = full_name[len(full_name) - 1]
                    if check[0] == self.name:
                        return check
                else:
                    check = line.split(';')
                    if check[0] == self.name:
                        return check
        else:
            for line in out.splitlines():
                if self.is_freebsd:
                    check = line.split()
                    if check[0] == self.name:
                        return check
                else:
                    check = line.split(';')
                    if check[0] == self.name:
                        return check
        return None

    def exists(self):
        rc, out, dummy = self._beadm_list()
        if rc == 0:
            if self._find_be_by_name(out):
                return True
            else:
                return False
        else:
            return False

    def is_activated(self):
        rc, out, dummy = self._beadm_list()
        if rc == 0:
            line = self._find_be_by_name(out)
            if line is None:
                return False
            if self.is_freebsd:
                if 'R' in line[1]:
                    return True
            elif 'R' in line[2]:
                return True
        return False

    def activate_be(self):
        cmd = [self.module.get_bin_path('beadm'), 'activate', self.name]
        return self.module.run_command(cmd)

    def create_be(self):
        cmd = [self.module.get_bin_path('beadm'), 'create']
        if self.snapshot:
            cmd.extend(['-e', self.snapshot])
        if not self.is_freebsd:
            if self.description:
                cmd.extend(['-d', self.description])
            if self.options:
                cmd.extend(['-o', self.options])
        cmd.append(self.name)
        return self.module.run_command(cmd)

    def destroy_be(self):
        cmd = [self.module.get_bin_path('beadm'), 'destroy', '-F', self.name]
        return self.module.run_command(cmd)

    def is_mounted(self):
        rc, out, dummy = self._beadm_list()
        if rc == 0:
            line = self._find_be_by_name(out)
            if line is None:
                return False
            if self.is_freebsd:
                if line[2] != '-' and line[2] != '/':
                    return True
            elif line[3]:
                return True
        return False

    def mount_be(self):
        cmd = [self.module.get_bin_path('beadm'), 'mount', self.name]
        if self.mountpoint:
            cmd.append(self.mountpoint)
        return self.module.run_command(cmd)

    def unmount_be(self):
        cmd = [self.module.get_bin_path('beadm'), 'unmount']
        if self.force:
            cmd.append('-f')
        cmd.append(self.name)
        return self.module.run_command(cmd)