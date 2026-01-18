from __future__ import absolute_import, division, print_function
import os
from tempfile import TemporaryFile, NamedTemporaryFile
from select import select
from abc import ABCMeta, abstractmethod
import tarfile
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.exceptions import (
from ansible.module_utils._text import to_native
class K8SCopyFromPod(K8SCopy):
    """
    Copy files/directory from Pod into local filesystem
    """

    def __init__(self, module, client):
        super(K8SCopyFromPod, self).__init__(module, client)
        self.is_remote_path_dir = None
        self.files_to_copy = []
        self._shellname = None

    @property
    def pod_shell(self):
        if self._shellname is None:
            for s in ('/bin/sh', '/bin/bash'):
                error, out, err = self._run_from_pod(s)
                if error.get('status') == 'Success':
                    self._shellname = s
                    break
        return self._shellname

    def listfiles_with_find(self, path):
        find_cmd = ['find', path, '-type', 'f']
        error, files, err = self._run_from_pod(cmd=find_cmd)
        if error.get('status') != 'Success':
            self.module.fail_json(msg=error.get('message'))
        return files

    def listfile_with_echo(self, path):
        echo_cmd = [self.pod_shell, '-c', 'echo {path}/* {path}/.*'.format(path=path.translate(str.maketrans({' ': '\\ '})))]
        error, out, err = self._run_from_pod(cmd=echo_cmd)
        if error.get('status') != 'Success':
            self.module.fail_json(msg=error.get('message'))
        files = []
        if out:
            output = out[0] + ' '
            files = [os.path.join(path, p[:-1]) for p in output.split(f'{path}/') if p and p[:-1] not in ('.', '..')]
        result = []
        for f in files:
            is_dir, err = self.is_directory_path_from_pod(f)
            if err:
                continue
            if not is_dir:
                result.append(f)
                continue
            result += self.listfile_with_echo(f)
        return result

    def list_remote_files(self):
        """
        This method will check if the remote path is a dir or file
        if it is a directory the file list will be updated accordingly
        """
        is_dir, error = self.is_directory_path_from_pod(self.remote_path)
        if error:
            self.module.fail_json(msg=error)
        if not is_dir:
            return [self.remote_path]
        else:
            executables = dict(find=self.listfiles_with_find, echo=self.listfile_with_echo)
            for item in executables:
                error, out, err = self._run_from_pod(item)
                if error.get('status') == 'Success':
                    return executables.get(item)(self.remote_path)

    def read(self):
        self.stdout = None
        self.stderr = None
        if self.response.is_open():
            if not self.response.sock.connected:
                self.response._connected = False
            else:
                ret, out, err = select((self.response.sock.sock,), (), (), 0)
                if ret:
                    code, frame = self.response.sock.recv_data_frame(True)
                    if code == ABNF.OPCODE_CLOSE:
                        self.response._connected = False
                    elif code in (ABNF.OPCODE_BINARY, ABNF.OPCODE_TEXT) and len(frame.data) > 1:
                        channel = frame.data[0]
                        content = frame.data[1:]
                        if content:
                            if channel == STDOUT_CHANNEL:
                                self.stdout = content
                            elif channel == STDERR_CHANNEL:
                                self.stderr = content.decode('utf-8', 'replace')

    def copy(self):
        is_remote_path_dir = len(self.files_to_copy) > 1 or self.files_to_copy[0] != self.remote_path
        relpath_start = self.remote_path
        if is_remote_path_dir and os.path.isdir(self.local_path):
            relpath_start = os.path.dirname(self.remote_path)
        if not self.check_mode:
            for remote_file in self.files_to_copy:
                dest_file = self.local_path
                if is_remote_path_dir:
                    dest_file = os.path.join(self.local_path, os.path.relpath(remote_file, start=relpath_start))
                    os.makedirs(os.path.dirname(dest_file), exist_ok=True)
                pod_command = ['cat', remote_file]
                self.response = stream(self.api_instance.connect_get_namespaced_pod_exec, self.name, self.namespace, command=pod_command, stderr=True, stdin=True, stdout=True, tty=False, _preload_content=False, **self.container_arg)
                errors = []
                with open(dest_file, 'wb') as fh:
                    while self.response._connected:
                        self.read()
                        if self.stdout:
                            fh.write(self.stdout)
                        if self.stderr:
                            errors.append(self.stderr)
                if errors:
                    self.module.fail_json(msg='Failed to copy file from Pod: {0}'.format(''.join(errors)))
        self.module.exit_json(changed=True, result='{0} successfully copied locally into {1}'.format(self.remote_path, self.local_path))

    def run(self):
        self.files_to_copy = self.list_remote_files()
        if self.files_to_copy == []:
            self.module.exit_json(changed=False, warning="No file found from directory '{0}' into remote Pod.".format(self.remote_path))
        self.copy()