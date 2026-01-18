from __future__ import (absolute_import, division, print_function)
import json
import os
import os.path
import stat
import tempfile
import traceback
from ansible import constants as C
from ansible.errors import AnsibleError, AnsibleFileNotFound
from ansible.module_utils.basic import FILE_COMMON_ARGUMENTS
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.plugins.action import ActionBase
from ansible.utils.hashing import checksum
def _ensure_invocation(self, result):
    if 'invocation' not in result:
        if self._play_context.no_log:
            result['invocation'] = 'CENSORED: no_log is set'
        else:
            result['invocation'] = self._task.args.copy()
            result['invocation']['module_args'] = self._task.args.copy()
    if isinstance(result['invocation'], dict):
        if 'content' in result['invocation']:
            result['invocation']['content'] = 'CENSORED: content is a no_log parameter'
        if result['invocation'].get('module_args', {}).get('content') is not None:
            result['invocation']['module_args']['content'] = 'VALUE_SPECIFIED_IN_NO_LOG_PARAMETER'
    return result