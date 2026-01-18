from __future__ import absolute_import, division, print_function
import hashlib
import os
import re
from ansible.module_utils._text import to_bytes, to_text
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.network import (
class FilePull(FileCopy):

    def __init__(self, module):
        super(FilePull, self).__init__(module)
        self.result = {}

    def mkdir(self, directory):
        local_dir_root = '/'
        dir_array = directory.split('/')
        for each in dir_array:
            if each:
                mkdir_cmd = 'mkdir ' + local_dir_root + each
                self._connection.run_commands(mkdir_cmd)
                local_dir_root += each + '/'
        return local_dir_root

    def copy_file_from_remote(self, local, local_file_directory, file_system):
        cmdroot = 'copy ' + self._module.params['file_pull_protocol'] + '://'
        ruser = self._module.params['remote_scp_server_user'] + '@'
        rserver = self._module.params['remote_scp_server']
        rserverpassword = self._module.params['remote_scp_server_password']
        rfile = self._module.params['remote_file'] + ' '
        if not rfile.startswith('/'):
            rfile = '/' + rfile
        if not self._platform.startswith('DS-') and 'MDS' not in self._model:
            vrf = ' vrf ' + self._module.params['vrf']
        else:
            vrf = ''
        if self._module.params['file_pull_compact']:
            compact = ' compact '
        else:
            compact = ''
        if self._module.params['file_pull_kstack']:
            kstack = ' use-kstack '
        else:
            kstack = ''
        local_dir_root = '/'
        if local_file_directory:
            local_dir_root = self.mkdir(local_file_directory)
        copy_cmd = cmdroot + ruser + rserver + rfile + file_system + local_dir_root + local + compact + vrf + kstack
        self.result['copy_cmd'] = copy_cmd
        pulled = self._connection.pull_file(command=copy_cmd, remotepassword=rserverpassword)
        if pulled:
            self.result['transfer_status'] = 'Received: File copied/pulled to nxos device from remote scp server.'
        else:
            self.result['failed'] = True

    def run(self):
        self.result['failed'] = False
        remote_file = self._module.params['remote_file']
        local_file = self._module.params['local_file'] or remote_file.split('/')[-1]
        file_system = self._module.params['file_system']
        local_file_dir = self._module.params['local_file_directory']
        if not self._module.check_mode:
            self.copy_file_from_remote(local_file, local_file_dir, file_system)
        self.result['remote_file'] = remote_file
        if local_file_dir:
            dir = local_file_dir
        else:
            dir = ''
        self.result['local_file'] = file_system + dir + '/' + local_file
        self.result['remote_scp_server'] = self._module.params['remote_scp_server']
        self.result['file_system'] = self._module.params['file_system']
        if not self.result['failed']:
            self.result['changed'] = True
        return self.result