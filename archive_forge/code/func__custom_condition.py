from __future__ import absolute_import, division, print_function
import base64
import time
import os
import traceback
import sys
import hashlib
from datetime import datetime
from tempfile import NamedTemporaryFile
from ansible_collections.kubernetes.core.plugins.module_utils.version import (
from ansible_collections.kubernetes.core.plugins.module_utils.args_common import (
from ansible_collections.kubernetes.core.plugins.module_utils.hashes import (
from ansible_collections.kubernetes.core.plugins.module_utils.selector import (
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import iteritems, string_types
from ansible.module_utils._text import to_native, to_bytes, to_text
from ansible.module_utils.common.dict_transformations import dict_merge
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.urls import Request
def _custom_condition(resource):
    if not resource.status or not resource.status.conditions:
        return False
    match = [x for x in resource.status.conditions if x.type == condition['type']]
    if not match:
        return False
    match = match[0]
    if match.status == 'Unknown':
        if match.status == condition['status']:
            if 'reason' not in condition:
                return True
            if condition['reason']:
                return match.reason == condition['reason']
        return False
    status = True if match.status == 'True' else False
    if status == boolean(condition['status'], strict=False):
        if condition.get('reason'):
            return match.reason == condition['reason']
        return True
    return False