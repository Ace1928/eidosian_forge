from __future__ import absolute_import, division, print_function
import hashlib
import os
import re
from ansible.module_utils._text import to_bytes, to_text
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.network import (
class FilePush(FileCopy):

    def __init__(self, module):
        super(FilePush, self).__init__(module)
        self.result = {}

    def md5sum_check(self, dst, file_system):
        command = 'show file {0}{1} md5sum'.format(file_system, dst)
        remote_filehash = self._connection.run_commands(command)[0]
        remote_filehash = to_bytes(remote_filehash, errors='surrogate_or_strict')
        local_file = self._module.params['local_file']
        try:
            with open(local_file, 'rb') as f:
                filecontent = f.read()
        except (OSError, IOError) as exc:
            self._module.fail_json('Error reading the file: {0}'.format(to_text(exc)))
        filecontent = to_bytes(filecontent, errors='surrogate_or_strict')
        local_filehash = hashlib.md5(filecontent).hexdigest()
        decoded_rhash = remote_filehash.decode('UTF-8')
        if local_filehash == decoded_rhash:
            return True
        else:
            return False

    def remote_file_exists(self, remote_file, file_system):
        command = 'dir {0}/{1}'.format(file_system, remote_file)
        body = self._connection.run_commands(command)[0]
        if 'No such file' in body:
            return False
        else:
            return self.md5sum_check(remote_file, file_system)

    def get_flash_size(self, file_system):
        command = 'dir {0}'.format(file_system)
        body = self._connection.run_commands(command)[0]
        match = re.search('(\\d+) bytes free', body)
        if match:
            bytes_free = match.group(1)
            return int(bytes_free)
        match = re.search('No such file or directory', body)
        if match:
            self._module.fail_json('Invalid nxos filesystem {0}'.format(file_system))
        else:
            self._module.fail_json('Unable to determine size of filesystem {0}'.format(file_system))

    def enough_space(self, file, file_system):
        flash_size = self.get_flash_size(file_system)
        file_size = os.path.getsize(file)
        if file_size > flash_size:
            return False
        return True

    def transfer_file_to_device(self, remote_file):
        local_file = self._module.params['local_file']
        file_system = self._module.params['file_system']
        if not self.enough_space(local_file, file_system):
            self._module.fail_json('Could not transfer file. Not enough space on device.')
        frp = remote_file
        if not file_system.startswith('bootflash:'):
            frp = '{0}{1}'.format(file_system, remote_file)
        flp = os.path.join(os.path.abspath(local_file))
        try:
            self._connection.copy_file(source=flp, destination=frp, proto='scp', timeout=self._connection.get_option('persistent_command_timeout'))
            self.result['transfer_status'] = 'Sent: File copied to remote device.'
        except Exception as exc:
            self.result['failed'] = True
            self.result['msg'] = 'Exception received : %s' % exc

    def run(self):
        local_file = self._module.params['local_file']
        remote_file = self._module.params['remote_file'] or os.path.basename(local_file)
        file_system = self._module.params['file_system']
        if not os.path.isfile(local_file):
            self._module.fail_json('Local file {0} not found'.format(local_file))
        remote_file = remote_file or os.path.basename(local_file)
        remote_exists = self.remote_file_exists(remote_file, file_system)
        if not remote_exists:
            self.result['changed'] = True
            file_exists = False
        else:
            self.result['transfer_status'] = 'No Transfer: File already copied to remote device.'
            file_exists = True
        if not self._module.check_mode and (not file_exists):
            self.transfer_file_to_device(remote_file)
        self.result['local_file'] = local_file
        if remote_file is None:
            remote_file = os.path.basename(local_file)
        self.result['remote_file'] = remote_file
        self.result['file_system'] = file_system
        return self.result