from __future__ import absolute_import, division, print_function
import abc
import binascii
import traceback
from ansible.module_utils import six
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.crypto.basic import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.cryptography_support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.cryptography_crl import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.csr_info import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.common import ArgumentSpec
def _check_keyUsage(extensions):
    current_keyusage_ext = _find_extension(extensions, cryptography.x509.KeyUsage)
    if not self.keyUsage:
        return current_keyusage_ext is None
    elif current_keyusage_ext is None:
        return False
    params = cryptography_parse_key_usage_params(self.keyUsage)
    for param in params:
        if getattr(current_keyusage_ext.value, '_' + param) != params[param]:
            return False
    if current_keyusage_ext.critical != self.keyUsage_critical:
        return False
    return True