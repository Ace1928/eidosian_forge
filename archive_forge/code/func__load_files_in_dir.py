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
def _load_files_in_dir(self, root_dir, var_files):
    """ Load the found yml files and update/overwrite the dictionary.
        Args:
            root_dir (str): The base directory of the list of files that is being passed.
            var_files: (list): List of files to iterate over and load into a dictionary.

        Returns:
            Tuple (bool, str, dict)
        """
    results = dict()
    failed = False
    err_msg = ''
    for filename in var_files:
        stop_iter = False
        if self._task._role:
            if path.join(self._task._role._role_path, filename) == path.join(root_dir, 'vars', 'main.yml'):
                stop_iter = True
                continue
        filepath = path.join(root_dir, filename)
        if self.files_matching:
            if not self.matcher.search(filename):
                stop_iter = True
        if not stop_iter and (not failed):
            if self.ignore_unknown_extensions:
                if path.exists(filepath) and (not self._ignore_file(filename)) and self._is_valid_file_ext(filename):
                    failed, err_msg, loaded_data = self._load_files(filepath, validate_extensions=True)
                    if not failed:
                        results.update(loaded_data)
            elif path.exists(filepath) and (not self._ignore_file(filename)):
                failed, err_msg, loaded_data = self._load_files(filepath, validate_extensions=True)
                if not failed:
                    results.update(loaded_data)
    return (failed, err_msg, results)