from __future__ import (absolute_import, division, print_function)
import os
import time
import glob
from ansible.module_utils.six.moves.urllib.parse import urlsplit
from ansible.module_utils._text import to_text
from ansible_collections.community.network.plugins.action.ce import ActionModule as _ActionModule
def _get_working_path(self):
    cwd = self._loader.get_basedir()
    if self._task._role is not None:
        cwd = self._task._role._role_path
    return cwd