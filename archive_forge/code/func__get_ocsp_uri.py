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
def _get_ocsp_uri(self):
    try:
        ext = self.cert.extensions.get_extension_for_class(x509.AuthorityInformationAccess)
        for desc in ext.value:
            if desc.access_method == x509.oid.AuthorityInformationAccessOID.OCSP:
                if isinstance(desc.access_location, x509.UniformResourceIdentifier):
                    return desc.access_location.value
    except x509.ExtensionNotFound as dummy:
        pass
    return None