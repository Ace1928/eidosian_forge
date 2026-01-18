from __future__ import (absolute_import, division, print_function)
import base64
import json
import os
import os.path
import shutil
import tempfile
import traceback
import zipfile
from ansible import constants as C
from ansible.errors import AnsibleError, AnsibleFileNotFound
from ansible.module_utils._text import to_bytes, to_native, to_text
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.plugins.action import ActionBase
from ansible.utils.hashing import checksum
def _copy_single_file(self, local_file, dest, source_rel, dest_rel, task_vars, tmp, backup):
    if self._play_context.check_mode:
        module_return = dict(changed=True)
        return module_return
    tmp_src = self._connection._shell.join_path(tmp, 'source')
    self._transfer_file(local_file, tmp_src)
    copy_args = self._task.args.copy()
    copy_args.update(dict(dest=dest, src=tmp_src, _original_basename=source_rel, _copy_mode='single', backup=backup))
    copy_args.pop('content', None)
    copy_result = self._execute_module(module_name='ansible.windows.win_copy', module_args=copy_args, task_vars=task_vars)
    return copy_result