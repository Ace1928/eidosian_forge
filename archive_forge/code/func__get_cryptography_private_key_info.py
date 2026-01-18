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
def _get_cryptography_private_key_info(key, need_private_key_data=False):
    key_type, key_public_data = _get_cryptography_public_key_info(key.public_key())
    key_private_data = dict()
    if need_private_key_data:
        if isinstance(key, cryptography.hazmat.primitives.asymmetric.rsa.RSAPrivateKey):
            private_numbers = key.private_numbers()
            key_private_data['p'] = private_numbers.p
            key_private_data['q'] = private_numbers.q
            key_private_data['exponent'] = private_numbers.d
        elif isinstance(key, cryptography.hazmat.primitives.asymmetric.dsa.DSAPrivateKey):
            private_numbers = key.private_numbers()
            key_private_data['x'] = private_numbers.x
        elif isinstance(key, cryptography.hazmat.primitives.asymmetric.ec.EllipticCurvePrivateKey):
            private_numbers = key.private_numbers()
            key_private_data['multiplier'] = private_numbers.private_value
    return (key_type, key_public_data, key_private_data)