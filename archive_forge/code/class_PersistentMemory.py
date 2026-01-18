from __future__ import (absolute_import, division, print_function)
import json
import re
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib, human_to_bytes
class PersistentMemory(object):

    def __init__(self):
        module = AnsibleModule(argument_spec=dict(appdirect=dict(type='int'), appdirect_interleaved=dict(type='bool', default=True), memorymode=dict(type='int'), reserved=dict(type='int'), socket=dict(type='list', elements='dict', options=dict(id=dict(required=True, type='int'), appdirect=dict(required=True, type='int'), appdirect_interleaved=dict(type='bool', default=True), memorymode=dict(required=True, type='int'), reserved=dict(type='int'))), namespace=dict(type='list', elements='dict', options=dict(mode=dict(required=True, type='str', choices=['raw', 'sector', 'fsdax', 'devdax']), type=dict(type='str', choices=['pmem', 'blk']), size=dict(type='str'))), namespace_append=dict(type='bool', default=False)), required_together=(['appdirect', 'memorymode'],), required_one_of=(['appdirect', 'memorymode', 'socket', 'namespace'],), mutually_exclusive=(['appdirect', 'socket'], ['memorymode', 'socket'], ['appdirect', 'namespace'], ['memorymode', 'namespace'], ['socket', 'namespace'], ['appdirect', 'namespace_append'], ['memorymode', 'namespace_append'], ['socket', 'namespace_append']))
        if not HAS_XMLTODICT_LIBRARY:
            module.fail_json(msg=missing_required_lib('xmltodict'), exception=XMLTODICT_LIBRARY_IMPORT_ERROR)
        self.ipmctl_exec = module.get_bin_path('ipmctl', True)
        self.ndctl_exec = module.get_bin_path('ndctl', True)
        self.appdirect = module.params['appdirect']
        self.interleaved = module.params['appdirect_interleaved']
        self.memmode = module.params['memorymode']
        self.reserved = module.params['reserved']
        self.socket = module.params['socket']
        self.namespace = module.params['namespace']
        self.namespace_append = module.params['namespace_append']
        self.module = module
        self.changed = False
        self.result = []

    def pmem_run_command(self, command, returnCheck=True):
        cmd = [str(part) for part in command]
        self.module.log(msg='pmem_run_command: execute: %s' % cmd)
        rc, out, err = self.module.run_command(cmd)
        self.module.log(msg='pmem_run_command: result: %s' % out)
        if returnCheck and rc != 0:
            self.module.fail_json(msg='Error while running: %s' % cmd, rc=rc, out=out, err=err)
        return out

    def pmem_run_ipmctl(self, command, returnCheck=True):
        command = [self.ipmctl_exec] + command
        return self.pmem_run_command(command, returnCheck)

    def pmem_run_ndctl(self, command, returnCheck=True):
        command = [self.ndctl_exec] + command
        return self.pmem_run_command(command, returnCheck)

    def pmem_is_dcpmm_installed(self):
        command = ['show', '-system', '-capabilities']
        return self.pmem_run_ipmctl(command)

    def pmem_get_region_align_size(self, region):
        aligns = []
        for rg in region:
            if rg['align'] not in aligns:
                aligns.append(rg['align'])
        return aligns

    def pmem_get_available_region_size(self, region):
        available_size = []
        for rg in region:
            available_size.append(rg['available_size'])
        return available_size

    def pmem_get_available_region_type(self, region):
        types = []
        for rg in region:
            if rg['type'] not in types:
                types.append(rg['type'])
        return types

    def pmem_argument_check(self):

        def namespace_check(self):
            command = ['list', '-R']
            out = self.pmem_run_ndctl(command)
            if not out:
                return 'Available region(s) is not in this system.'
            region = json.loads(out)
            aligns = self.pmem_get_region_align_size(region)
            if len(aligns) != 1:
                return 'Not supported the regions whose alignment size is different.'
            available_size = self.pmem_get_available_region_size(region)
            types = self.pmem_get_available_region_type(region)
            for ns in self.namespace:
                if ns['size']:
                    try:
                        size_byte = human_to_bytes(ns['size'])
                    except ValueError:
                        return 'The format of size: NNN TB|GB|MB|KB|T|G|M|K|B'
                    if size_byte % aligns[0] != 0:
                        return 'size: %s should be align with %d' % (ns['size'], aligns[0])
                    is_space_enough = False
                    for i, avail in enumerate(available_size):
                        if avail > size_byte:
                            available_size[i] -= size_byte
                            is_space_enough = True
                            break
                    if is_space_enough is False:
                        return 'There is not available region for size: %s' % ns['size']
                    ns['size_byte'] = size_byte
                elif len(self.namespace) != 1:
                    return 'size option is required to configure multiple namespaces'
                if ns['type'] not in types:
                    return 'type %s is not supported in this system. Supported type: %s' % (ns['type'], types)
            return None

        def percent_check(self, appdirect, memmode, reserved=None):
            if appdirect is None or (appdirect < 0 or appdirect > 100):
                return 'appdirect percent should be from 0 to 100.'
            if memmode is None or (memmode < 0 or memmode > 100):
                return 'memorymode percent should be from 0 to 100.'
            if reserved is None:
                if appdirect + memmode > 100:
                    return 'Total percent should be less equal 100.'
            else:
                if reserved < 0 or reserved > 100:
                    return 'reserved percent should be from 0 to 100.'
                if appdirect + memmode + reserved != 100:
                    return 'Total percent should be 100.'

        def socket_id_check(self):
            command = ['show', '-o', 'nvmxml', '-socket']
            out = self.pmem_run_ipmctl(command)
            sockets_dict = xmltodict.parse(out, dict_constructor=dict)['SocketList']['Socket']
            socket_ids = []
            for sl in sockets_dict:
                socket_ids.append(int(sl['SocketID'], 16))
            for skt in self.socket:
                if skt['id'] not in socket_ids:
                    return 'Invalid socket number: %d' % skt['id']
            return None
        if self.namespace:
            return namespace_check(self)
        elif self.socket is None:
            return percent_check(self, self.appdirect, self.memmode, self.reserved)
        else:
            ret = socket_id_check(self)
            if ret is not None:
                return ret
            for skt in self.socket:
                ret = percent_check(self, skt['appdirect'], skt['memorymode'], skt['reserved'])
                if ret is not None:
                    return ret
            return None

    def pmem_remove_namespaces(self):
        command = ['list', '-N']
        out = self.pmem_run_ndctl(command)
        if not out:
            return
        namespaces = json.loads(out)
        for ns in namespaces:
            command = ['disable-namespace', ns['dev']]
            self.pmem_run_ndctl(command)
            command = ['destroy-namespace', ns['dev']]
            self.pmem_run_ndctl(command)
        return

    def pmem_delete_goal(self):
        command = ['delete', '-goal']
        self.pmem_run_ipmctl(command)

    def pmem_init_env(self):
        if self.namespace is None or (self.namespace and self.namespace_append is False):
            self.pmem_remove_namespaces()
        if self.namespace is None:
            self.pmem_delete_goal()

    def pmem_get_capacity(self, skt=None):
        command = ['show', '-d', 'Capacity', '-u', 'B', '-o', 'nvmxml', '-dimm']
        if skt:
            command += ['-socket', skt['id']]
        out = self.pmem_run_ipmctl(command)
        dimm_list = xmltodict.parse(out, dict_constructor=dict)['DimmList']['Dimm']
        capacity = 0
        for entry in dimm_list:
            for key, v in entry.items():
                if key == 'Capacity':
                    capacity += int(v.split()[0])
        return capacity

    def pmem_create_memory_allocation(self, skt=None):

        def build_ipmctl_creation_opts(self, skt=None):
            ipmctl_opts = []
            if skt:
                appdirect = skt['appdirect']
                memmode = skt['memorymode']
                reserved = skt['reserved']
                socket_id = skt['id']
                ipmctl_opts += ['-socket', socket_id]
            else:
                appdirect = self.appdirect
                memmode = self.memmode
                reserved = self.reserved
            if reserved is None:
                res = 100 - memmode - appdirect
                ipmctl_opts += ['memorymode=%d' % memmode, 'reserved=%d' % res]
            else:
                ipmctl_opts += ['memorymode=%d' % memmode, 'reserved=%d' % reserved]
            if self.interleaved:
                ipmctl_opts += ['PersistentMemoryType=AppDirect']
            else:
                ipmctl_opts += ['PersistentMemoryType=AppDirectNotInterleaved']
            return ipmctl_opts

        def is_allocation_good(self, ipmctl_out, command):
            warning = re.compile('WARNING')
            error = re.compile('.*Error.*')
            ignore_error = re.compile('Do you want to continue? [y/n] Error: Invalid data input.')
            errmsg = ''
            rc = True
            for line in ipmctl_out.splitlines():
                if warning.match(line):
                    errmsg = '%s (command: %s)' % (line, command)
                    rc = False
                    break
                elif error.match(line):
                    if not ignore_error:
                        errmsg = '%s (command: %s)' % (line, command)
                        rc = False
                        break
            return (rc, errmsg)

        def get_allocation_result(self, goal, skt=None):
            ret = {'appdirect': 0, 'memorymode': 0}
            if skt:
                ret['socket'] = skt['id']
            out = xmltodict.parse(goal, dict_constructor=dict)['ConfigGoalList']['ConfigGoal']
            for entry in out:
                if skt and skt['id'] != int(entry['SocketID'], 16):
                    continue
                for key, v in entry.items():
                    if key == 'MemorySize':
                        ret['memorymode'] += int(v.split()[0])
                    elif key == 'AppDirect1Size' or key == 'AapDirect2Size':
                        ret['appdirect'] += int(v.split()[0])
            capacity = self.pmem_get_capacity(skt)
            ret['reserved'] = capacity - ret['appdirect'] - ret['memorymode']
            return ret
        reboot_required = False
        ipmctl_opts = build_ipmctl_creation_opts(self, skt)
        command = ['create', '-goal'] + ipmctl_opts
        out = self.pmem_run_ipmctl(command, returnCheck=False)
        rc, errmsg = is_allocation_good(self, out, command)
        if rc is False:
            return (reboot_required, {}, errmsg)
        command = ['create', '-u', 'B', '-o', 'nvmxml', '-force', '-goal'] + ipmctl_opts
        goal = self.pmem_run_ipmctl(command)
        ret = get_allocation_result(self, goal, skt)
        reboot_required = True
        return (reboot_required, ret, '')

    def pmem_config_namespaces(self, namespace):
        command = ['create-namespace', '-m', namespace['mode']]
        if namespace['type']:
            command += ['-t', namespace['type']]
        if 'size_byte' in namespace:
            command += ['-s', namespace['size_byte']]
        self.pmem_run_ndctl(command)
        return None