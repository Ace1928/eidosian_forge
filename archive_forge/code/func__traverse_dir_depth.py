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
def _traverse_dir_depth(self):
    """ Recursively iterate over a directory and sort the files in
            alphabetical order. Do not iterate pass the set depth.
            The default depth is unlimited.
        """
    sorted_walk = list(walk(self.source_dir, onerror=self._log_walk, followlinks=True))
    sorted_walk.sort(key=lambda x: x[0])
    for current_root, current_dir, current_files in sorted_walk:
        current_depth = len(pathlib.Path(current_root).relative_to(self.source_dir).parts) + 1
        if self.depth != 0 and current_depth > self.depth:
            continue
        current_files.sort()
        yield (current_root, current_files)