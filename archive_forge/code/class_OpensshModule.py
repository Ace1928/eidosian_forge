from __future__ import absolute_import, division, print_function
import abc
import os
import stat
import traceback
from ansible.module_utils import six
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.crypto.plugins.module_utils.openssh.utils import (
@six.add_metaclass(abc.ABCMeta)
class OpensshModule(object):

    def __init__(self, module):
        self.module = module
        self.changed = False
        self.check_mode = self.module.check_mode

    def execute(self):
        try:
            self._execute()
        except Exception as e:
            self.module.fail_json(msg='unexpected error occurred: %s' % to_native(e), exception=traceback.format_exc())
        self.module.exit_json(**self.result)

    @abc.abstractmethod
    def _execute(self):
        pass

    @property
    def result(self):
        result = self._result
        result['changed'] = self.changed
        if self.module._diff:
            result['diff'] = self.diff
        return result

    @property
    @abc.abstractmethod
    def _result(self):
        pass

    @property
    @abc.abstractmethod
    def diff(self):
        pass

    @staticmethod
    def skip_if_check_mode(f):

        def wrapper(self, *args, **kwargs):
            if not self.check_mode:
                f(self, *args, **kwargs)
        return wrapper

    @staticmethod
    def trigger_change(f):

        def wrapper(self, *args, **kwargs):
            f(self, *args, **kwargs)
            self.changed = True
        return wrapper

    def _check_if_base_dir(self, path):
        base_dir = os.path.dirname(path) or '.'
        if not os.path.isdir(base_dir):
            self.module.fail_json(name=base_dir, msg='The directory %s does not exist or the file is not a directory' % base_dir)

    def _get_ssh_version(self):
        ssh_bin = self.module.get_bin_path('ssh')
        if not ssh_bin:
            return ''
        return parse_openssh_version(self.module.run_command([ssh_bin, '-V', '-q'], check_rc=True)[2].strip())

    @_restore_all_on_failure
    def _safe_secure_move(self, sources_and_destinations):
        """Moves a list of files from 'source' to 'destination' and restores 'destination' from backup upon failure.
           If 'destination' does not already exist, then 'source' permissions are preserved to prevent
           exposing protected data ('atomic_move' uses the 'destination' base directory mask for
           permissions if 'destination' does not already exists).
        """
        for source, destination in sources_and_destinations:
            if os.path.exists(destination):
                self.module.atomic_move(source, destination)
            else:
                self.module.preserved_copy(source, destination)

    def _update_permissions(self, path):
        file_args = self.module.load_file_common_arguments(self.module.params)
        file_args['path'] = path
        if not self.module.check_file_absent_if_check_mode(path):
            self.changed = self.module.set_fs_attributes_if_different(file_args, self.changed)
        else:
            self.changed = True