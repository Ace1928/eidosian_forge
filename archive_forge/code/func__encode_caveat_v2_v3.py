import base64
import json
from ._versions import (VERSION_1, VERSION_2, VERSION_3)
from ._third_party import legacy_namespace, ThirdPartyCaveatInfo
from ._keys import PublicKey
from ._error import VerificationError
import macaroonbakery.checkers as checkers
import nacl.public
import six
def _encode_caveat_v2_v3(version, condition, root_key, third_party_pub_key, key, ns):
    """Create a version 2 or version 3 third-party caveat.

    The format has the following packed binary fields (note
    that all fields up to and including the nonce are the same
    as the v2 format):

        version 2 or 3 [1 byte]
        first 4 bytes of third-party Curve25519 public key [4 bytes]
        first-party Curve25519 public key [32 bytes]
        nonce [24 bytes]
        encrypted secret part [rest of message]

    The encrypted part encrypts the following fields
    with box.Seal:

        version 2 or 3 [1 byte]
        length of root key [n: uvarint]
        root key [n bytes]
        length of encoded namespace [n: uvarint] (Version 3 only)
        encoded namespace [n bytes] (Version 3 only)
        condition [rest of encrypted part]
    """
    ns_data = bytearray()
    if version >= VERSION_3:
        ns_data = ns.serialize_text()
    data = bytearray()
    data.append(version)
    data.extend(third_party_pub_key.serialize(raw=True)[:_PUBLIC_KEY_PREFIX_LEN])
    data.extend(key.public_key.serialize(raw=True)[:])
    secret = _encode_secret_part_v2_v3(version, condition, root_key, ns_data)
    box = nacl.public.Box(key.key, third_party_pub_key.key)
    encrypted = box.encrypt(secret)
    nonce = encrypted[0:nacl.public.Box.NONCE_SIZE]
    encrypted = encrypted[nacl.public.Box.NONCE_SIZE:]
    data.extend(nonce[:])
    data.extend(encrypted)
    return bytes(data)