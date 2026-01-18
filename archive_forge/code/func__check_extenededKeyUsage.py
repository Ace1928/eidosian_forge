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
def _check_extenededKeyUsage(extensions):
    current_usages_ext = _find_extension(extensions, cryptography.x509.ExtendedKeyUsage)
    current_usages = [str(usage) for usage in current_usages_ext.value] if current_usages_ext else []
    usages = [str(cryptography_name_to_oid(usage)) for usage in self.extendedKeyUsage] if self.extendedKeyUsage else []
    if set(current_usages) != set(usages):
        return False
    if usages:
        if current_usages_ext.critical != self.extendedKeyUsage_critical:
            return False
    return True