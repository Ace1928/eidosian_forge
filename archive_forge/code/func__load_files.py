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
def _load_files(self, filename, validate_extensions=False):
    """ Loads a file and converts the output into a valid Python dict.
        Args:
            filename (str): The source file.

        Returns:
            Tuple (bool, str, dict)
        """
    results = dict()
    failed = False
    err_msg = ''
    if validate_extensions and (not self._is_valid_file_ext(filename)):
        failed = True
        err_msg = '{0} does not have a valid extension: {1}'.format(to_native(filename), ', '.join(self.valid_extensions))
    else:
        b_data, show_content = self._loader._get_file_contents(filename)
        data = to_text(b_data, errors='surrogate_or_strict')
        self.show_content = show_content
        data = self._loader.load(data, file_name=filename, show_content=show_content)
        if not data:
            data = dict()
        if not isinstance(data, dict):
            failed = True
            err_msg = '{0} must be stored as a dictionary/hash'.format(to_native(filename))
        else:
            self.included_files.append(filename)
            results.update(data)
    return (failed, err_msg, results)