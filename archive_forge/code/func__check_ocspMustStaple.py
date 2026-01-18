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
def _check_ocspMustStaple(extensions):
    try:
        tlsfeature_ext = _find_extension(extensions, cryptography.x509.TLSFeature)
        has_tlsfeature = True
    except AttributeError as dummy:
        tlsfeature_ext = next((ext for ext in extensions if ext.value.oid == CRYPTOGRAPHY_MUST_STAPLE_NAME), None)
        has_tlsfeature = False
    if self.ocspMustStaple:
        if not tlsfeature_ext or tlsfeature_ext.critical != self.ocspMustStaple_critical:
            return False
        if has_tlsfeature:
            return cryptography.x509.TLSFeatureType.status_request in tlsfeature_ext.value
        else:
            return tlsfeature_ext.value.value == CRYPTOGRAPHY_MUST_STAPLE_VALUE
    else:
        return tlsfeature_ext is None