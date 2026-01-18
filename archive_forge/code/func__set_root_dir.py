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
def _set_root_dir(self):
    if self._task._role:
        if self.source_dir.split('/')[0] == 'vars':
            path_to_use = path.join(self._task._role._role_path, self.source_dir)
            if path.exists(path_to_use):
                self.source_dir = path_to_use
        else:
            path_to_use = path.join(self._task._role._role_path, 'vars', self.source_dir)
            self.source_dir = path_to_use
    elif hasattr(self._task._ds, '_data_source'):
        current_dir = '/'.join(self._task._ds._data_source.split('/')[:-1])
        self.source_dir = path.join(current_dir, self.source_dir)