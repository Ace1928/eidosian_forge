from __future__ import absolute_import, division, print_function
import abc
import traceback
from ansible.module_utils import six
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.text.converters import to_native, to_bytes
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.crypto.basic import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.math import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.publickey_info import (
def _is_key_consistent(self, key_public_data, key_private_data):
    return _is_cryptography_key_consistent(self.key, key_public_data, key_private_data, warn_func=self.module.warn)