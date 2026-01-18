from __future__ import absolute_import, division, print_function
import abc
import os
import stat
import traceback
from ansible.module_utils import six
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.crypto.plugins.module_utils.openssh.utils import (
def _check_if_base_dir(self, path):
    base_dir = os.path.dirname(path) or '.'
    if not os.path.isdir(base_dir):
        self.module.fail_json(name=base_dir, msg='The directory %s does not exist or the file is not a directory' % base_dir)