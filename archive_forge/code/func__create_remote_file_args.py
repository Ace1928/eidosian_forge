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
def _create_remote_file_args(module_args):
    """remove keys that are not relevant to file"""
    return dict(((k, v) for k, v in module_args.items() if k in REAL_FILE_ARGS))