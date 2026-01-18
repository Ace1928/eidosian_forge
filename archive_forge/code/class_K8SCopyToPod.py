from __future__ import absolute_import, division, print_function
import os
from tempfile import TemporaryFile, NamedTemporaryFile
from select import select
from abc import ABCMeta, abstractmethod
import tarfile
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.exceptions import (
from ansible.module_utils._text import to_native
class K8SCopyToPod(K8SCopy):
    """
    Copy files/directory from local filesystem into remote Pod
    """

    def __init__(self, module, client):
        super(K8SCopyToPod, self).__init__(module, client)
        self.files_to_copy = list()

    def close_temp_file(self):
        if self.named_temp_file:
            self.named_temp_file.close()

    def run(self):
        dest_file = self.remote_path.rstrip('/')
        src_file = self.local_path
        self.named_temp_file = None
        if self.content:
            self.named_temp_file = NamedTemporaryFile(mode='w')
            self.named_temp_file.write(self.content)
            self.named_temp_file.flush()
            src_file = self.named_temp_file.name
        else:
            if not os.path.exists(self.local_path):
                self.module.fail_json(msg='{0} does not exist in local filesystem'.format(self.local_path))
            if not os.access(self.local_path, os.R_OK):
                self.module.fail_json(msg='{0} not readable'.format(self.local_path))
        is_dir, err = self.is_directory_path_from_pod(self.remote_path, failed_if_not_exists=False)
        if err:
            self.module.fail_json(msg=err)
        if is_dir:
            if self.content:
                self.module.fail_json(msg='When content is specified, remote path should not be an existing directory')
            else:
                dest_file = os.path.join(dest_file, os.path.basename(src_file))
        if not self.check_mode:
            if self.no_preserve:
                tar_command = ['tar', '--no-same-permissions', '--no-same-owner', '-xmf', '-']
            else:
                tar_command = ['tar', '-xmf', '-']
            if dest_file.startswith('/'):
                tar_command.extend(['-C', '/'])
            response = stream(self.api_instance.connect_get_namespaced_pod_exec, self.name, self.namespace, command=tar_command, stderr=True, stdin=True, stdout=True, tty=False, _preload_content=False, **self.container_arg)
            with TemporaryFile() as tar_buffer:
                with tarfile.open(fileobj=tar_buffer, mode='w') as tar:
                    tar.add(src_file, dest_file)
                tar_buffer.seek(0)
                commands = []
                size = 1024 * 1024
                while True:
                    data = tar_buffer.read(size)
                    if not data:
                        break
                    commands.append(data)
                stderr, stdout = ([], [])
                while response.is_open():
                    if response.peek_stdout():
                        stdout.append(response.read_stdout().rstrip('\n'))
                    if response.peek_stderr():
                        stderr.append(response.read_stderr().rstrip('\n'))
                    if commands:
                        cmd = commands.pop(0)
                        response.write_stdin(cmd)
                    else:
                        break
                response.close()
                if stderr:
                    self.close_temp_file()
                    self.module.fail_json(command=tar_command, msg='Failed to copy local file/directory into Pod due to: {0}'.format(''.join(stderr)))
            self.close_temp_file()
        if self.content:
            self.module.exit_json(changed=True, result='Content successfully copied into {0} on remote Pod'.format(self.remote_path))
        self.module.exit_json(changed=True, result='{0} successfully copied into remote Pod into {1}'.format(self.local_path, self.remote_path))