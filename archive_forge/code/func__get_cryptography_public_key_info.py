from __future__ import absolute_import, division, print_function
import abc
import traceback
from ansible.module_utils import six
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.crypto.basic import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import (
def _get_cryptography_public_key_info(key):
    key_public_data = dict()
    if isinstance(key, cryptography.hazmat.primitives.asymmetric.rsa.RSAPublicKey):
        key_type = 'RSA'
        public_numbers = key.public_numbers()
        key_public_data['size'] = key.key_size
        key_public_data['modulus'] = public_numbers.n
        key_public_data['exponent'] = public_numbers.e
    elif isinstance(key, cryptography.hazmat.primitives.asymmetric.dsa.DSAPublicKey):
        key_type = 'DSA'
        parameter_numbers = key.parameters().parameter_numbers()
        public_numbers = key.public_numbers()
        key_public_data['size'] = key.key_size
        key_public_data['p'] = parameter_numbers.p
        key_public_data['q'] = parameter_numbers.q
        key_public_data['g'] = parameter_numbers.g
        key_public_data['y'] = public_numbers.y
    elif CRYPTOGRAPHY_HAS_X25519 and isinstance(key, cryptography.hazmat.primitives.asymmetric.x25519.X25519PublicKey):
        key_type = 'X25519'
    elif CRYPTOGRAPHY_HAS_X448 and isinstance(key, cryptography.hazmat.primitives.asymmetric.x448.X448PublicKey):
        key_type = 'X448'
    elif CRYPTOGRAPHY_HAS_ED25519 and isinstance(key, cryptography.hazmat.primitives.asymmetric.ed25519.Ed25519PublicKey):
        key_type = 'Ed25519'
    elif CRYPTOGRAPHY_HAS_ED448 and isinstance(key, cryptography.hazmat.primitives.asymmetric.ed448.Ed448PublicKey):
        key_type = 'Ed448'
    elif isinstance(key, cryptography.hazmat.primitives.asymmetric.ec.EllipticCurvePublicKey):
        key_type = 'ECC'
        public_numbers = key.public_numbers()
        key_public_data['curve'] = key.curve.name
        key_public_data['x'] = public_numbers.x
        key_public_data['y'] = public_numbers.y
        key_public_data['exponent_size'] = key.curve.key_size
    else:
        key_type = 'unknown ({0})'.format(type(key))
    return (key_type, key_public_data)