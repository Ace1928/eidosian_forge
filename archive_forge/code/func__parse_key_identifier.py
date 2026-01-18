from __future__ import absolute_import, division, print_function
import base64
import binascii
import datetime
import os
import traceback
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.acme.backends import (
from ansible_collections.community.crypto.plugins.module_utils.acme.certificates import (
from ansible_collections.community.crypto.plugins.module_utils.acme.errors import (
from ansible_collections.community.crypto.plugins.module_utils.acme.io import read_file
from ansible_collections.community.crypto.plugins.module_utils.acme.utils import nopad_b64
from ansible_collections.community.crypto.plugins.module_utils.crypto.math import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.cryptography_support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.pem import (
@staticmethod
def _parse_key_identifier(key_identifier, name, criterium_idx, module):
    if key_identifier:
        try:
            return binascii.unhexlify(key_identifier.replace(':', ''))
        except Exception:
            if criterium_idx is None:
                module.warn('Criterium has invalid {0} value. Ignoring criterium.'.format(name))
            else:
                module.warn('Criterium {0} in select_chain has invalid {1} value. Ignoring criterium.'.format(criterium_idx, name))
    return None