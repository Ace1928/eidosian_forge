from __future__ import absolute_import, division, print_function
import abc
import traceback
from ansible.module_utils import six
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.crypto.basic import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import (
def _get_public_key(self, binary):
    return self.key.public_bytes(serialization.Encoding.DER if binary else serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)