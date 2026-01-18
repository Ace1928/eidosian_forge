import base64
import json
from ._versions import (VERSION_1, VERSION_2, VERSION_3)
from ._third_party import legacy_namespace, ThirdPartyCaveatInfo
from ._keys import PublicKey
from ._error import VerificationError
import macaroonbakery.checkers as checkers
import nacl.public
import six
def _decode_secret_part_v2_v3(version, data):
    if len(data) < 1:
        raise VerificationError('secret part too short')
    got_version = six.byte2int(data[:1])
    data = data[1:]
    if version != got_version:
        raise VerificationError('unexpected secret part version, got {} want {}'.format(got_version, version))
    root_key_length, read = decode_uvarint(data)
    data = data[read:]
    root_key = data[:root_key_length]
    data = data[root_key_length:]
    if version >= VERSION_3:
        namespace_length, read = decode_uvarint(data)
        data = data[read:]
        ns_data = data[:namespace_length]
        data = data[namespace_length:]
        ns = checkers.deserialize_namespace(ns_data)
    else:
        ns = legacy_namespace()
    return (root_key, data, ns)