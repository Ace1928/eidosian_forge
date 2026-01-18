from __future__ import absolute_import, division, print_function
import abc
import binascii
import datetime
import traceback
from ansible.module_utils import six
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.cryptography_support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.publickey_info import (
def _get_ocsp_must_staple(self):
    try:
        try:
            tlsfeature_ext = self.cert.extensions.get_extension_for_class(x509.TLSFeature)
            value = cryptography.x509.TLSFeatureType.status_request in tlsfeature_ext.value
        except AttributeError:
            oid = x509.oid.ObjectIdentifier('1.3.6.1.5.5.7.1.24')
            tlsfeature_ext = self.cert.extensions.get_extension_for_oid(oid)
            value = tlsfeature_ext.value.value == b'0\x03\x02\x01\x05'
        return (value, tlsfeature_ext.critical)
    except cryptography.x509.ExtensionNotFound:
        return (None, False)