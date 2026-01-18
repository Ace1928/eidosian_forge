from __future__ import absolute_import, division, print_function
import abc
import os
from ansible.module_utils import six
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.text.converters import to_native, to_text, to_bytes
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.openssh.cryptography import (
from ansible_collections.community.crypto.plugins.module_utils.openssh.backends.common import (
from ansible_collections.community.crypto.plugins.module_utils.openssh.utils import (
def _private_key_exists(self):
    return os.path.exists(self.private_key_path)