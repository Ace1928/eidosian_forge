import base64
import json
from ._versions import (VERSION_1, VERSION_2, VERSION_3)
from ._third_party import legacy_namespace, ThirdPartyCaveatInfo
from ._keys import PublicKey
from ._error import VerificationError
import macaroonbakery.checkers as checkers
import nacl.public
import six
def _encode_secret_part_v2_v3(version, condition, root_key, ns):
    """Creates a version 2 or version 3 secret part of the third party
    caveat. The returned data is not encrypted.

    The format has the following packed binary fields:
    version 2 or 3 [1 byte]
    root key length [n: uvarint]
    root key [n bytes]
    namespace length [n: uvarint] (v3 only)
    namespace [n bytes] (v3 only)
    predicate [rest of message]
    """
    data = bytearray()
    data.append(version)
    encode_uvarint(len(root_key), data)
    data.extend(root_key)
    if version >= VERSION_3:
        encode_uvarint(len(ns), data)
        data.extend(ns)
    data.extend(condition.encode('utf-8'))
    return bytes(data)