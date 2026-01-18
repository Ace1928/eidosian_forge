from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
class Zfs(object):

    def __init__(self, module, name, properties):
        self.module = module
        self.name = name
        self.properties = properties
        self.changed = False
        self.zfs_cmd = module.get_bin_path('zfs', True)
        self.zpool_cmd = module.get_bin_path('zpool', True)
        self.pool = name.split('/')[0].split('@')[0]
        self.is_solaris = os.uname()[0] == 'SunOS'
        self.is_openzfs = self.check_openzfs()
        self.enhanced_sharing = self.check_enhanced_sharing()

    def check_openzfs(self):
        cmd = [self.zpool_cmd]
        cmd.extend(['get', 'version'])
        cmd.append(self.pool)
        rc, out, err = self.module.run_command(cmd, check_rc=True)
        version = out.splitlines()[-1].split()[2]
        if version == '-':
            return True
        if int(version) == 5000:
            return True
        return False

    def check_enhanced_sharing(self):
        if self.is_solaris and (not self.is_openzfs):
            cmd = [self.zpool_cmd]
            cmd.extend(['get', 'version'])
            cmd.append(self.pool)
            rc, out, err = self.module.run_command(cmd, check_rc=True)
            version = out.splitlines()[-1].split()[2]
            if int(version) >= 34:
                return True
        return False

    def exists(self):
        cmd = [self.zfs_cmd, 'list', '-t', 'all', self.name]
        rc, dummy, dummy = self.module.run_command(cmd)
        return rc == 0

    def create(self):
        if self.module.check_mode:
            self.changed = True
            return
        properties = self.properties
        origin = self.module.params.get('origin')
        cmd = [self.zfs_cmd]
        if '@' in self.name:
            action = 'snapshot'
        elif origin:
            action = 'clone'
        else:
            action = 'create'
        cmd.append(action)
        if action in ['create', 'clone']:
            cmd += ['-p']
        if properties:
            for prop, value in properties.items():
                if prop == 'volsize':
                    cmd += ['-V', value]
                elif prop == 'volblocksize':
                    cmd += ['-b', value]
                else:
                    cmd += ['-o', '%s=%s' % (prop, value)]
        if origin and action == 'clone':
            cmd.append(origin)
        cmd.append(self.name)
        self.module.run_command(cmd, check_rc=True)
        self.changed = True

    def destroy(self):
        if self.module.check_mode:
            self.changed = True
            return
        cmd = [self.zfs_cmd, 'destroy', '-R', self.name]
        self.module.run_command(cmd, check_rc=True)
        self.changed = True

    def set_property(self, prop, value):
        if self.module.check_mode:
            self.changed = True
            return
        cmd = [self.zfs_cmd, 'set', prop + '=' + str(value), self.name]
        self.module.run_command(cmd, check_rc=True)

    def set_properties_if_changed(self):
        diff = {'before': {'extra_zfs_properties': {}}, 'after': {'extra_zfs_properties': {}}}
        current_properties = self.get_current_properties()
        for prop, value in self.properties.items():
            current_value = current_properties.get(prop, None)
            if current_value != value:
                self.set_property(prop, value)
                diff['before']['extra_zfs_properties'][prop] = current_value
                diff['after']['extra_zfs_properties'][prop] = value
        if self.module.check_mode:
            return diff
        updated_properties = self.get_current_properties()
        for prop in self.properties:
            value = updated_properties.get(prop, None)
            if value is None:
                self.module.fail_json(msg='zfsprop was not present after being successfully set: %s' % prop)
            if current_properties.get(prop, None) != value:
                self.changed = True
            if prop in diff['after']['extra_zfs_properties']:
                diff['after']['extra_zfs_properties'][prop] = value
        return diff

    def get_current_properties(self):
        cmd = [self.zfs_cmd, 'get', '-H', '-p', '-o', 'property,value,source']
        if self.enhanced_sharing:
            cmd += ['-e']
        cmd += ['all', self.name]
        rc, out, err = self.module.run_command(cmd)
        properties = dict()
        for line in out.splitlines():
            prop, value, source = line.split('\t')
            if source in ('local', 'received', '-'):
                properties[prop] = value
        if self.enhanced_sharing:
            properties['sharenfs'] = properties.get('share.nfs', None)
            properties['sharesmb'] = properties.get('share.smb', None)
        return properties