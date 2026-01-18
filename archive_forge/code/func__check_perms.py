from __future__ import absolute_import, division, print_function
import abc
import datetime
import errno
import hashlib
import os
import re
from ansible.module_utils import six
from ansible.module_utils.common.text.converters import to_native, to_bytes
from ansible_collections.community.crypto.plugins.module_utils.crypto.pem import (
from .basic import (
def _check_perms(module):
    file_args = module.load_file_common_arguments(module.params)
    if module.check_file_absent_if_check_mode(file_args['path']):
        return False
    return not module.set_fs_attributes_if_different(file_args, False)