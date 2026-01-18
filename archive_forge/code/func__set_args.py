from __future__ import (absolute_import, division, print_function)
from os import path, walk
import re
import pathlib
import ansible.constants as C
from ansible.errors import AnsibleError
from ansible.module_utils.six import string_types
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.plugins.action import ActionBase
from ansible.utils.vars import combine_vars
def _set_args(self):
    """ Set instance variables based on the arguments that were passed """
    self.hash_behaviour = self._task.args.get('hash_behaviour', None)
    self.return_results_as_name = self._task.args.get('name', None)
    self.source_dir = self._task.args.get('dir', None)
    self.source_file = self._task.args.get('file', None)
    if not self.source_dir and (not self.source_file):
        self.source_file = self._task.args.get('_raw_params')
        if self.source_file:
            self.source_file = self.source_file.rstrip('\n')
    self.depth = self._task.args.get('depth', None)
    self.files_matching = self._task.args.get('files_matching', None)
    self.ignore_unknown_extensions = self._task.args.get('ignore_unknown_extensions', False)
    self.ignore_files = self._task.args.get('ignore_files', None)
    self.valid_extensions = self._task.args.get('extensions', self.VALID_FILE_EXTENSIONS)
    if isinstance(self.valid_extensions, string_types):
        self.valid_extensions = list(self.valid_extensions)
    if not isinstance(self.valid_extensions, list):
        raise AnsibleError('Invalid type for "extensions" option, it must be a list')