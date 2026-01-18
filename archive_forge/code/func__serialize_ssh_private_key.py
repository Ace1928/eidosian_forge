from __future__ import annotations
import binascii
import enum
import os
import re
import typing
import warnings
from base64 import encodebytes as _base64_encode
from dataclasses import dataclass
from cryptography import utils
from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import (
from cryptography.hazmat.primitives.asymmetric import utils as asym_utils
from cryptography.hazmat.primitives.ciphers import (
from cryptography.hazmat.primitives.serialization import (
def _serialize_ssh_private_key(private_key: SSHPrivateKeyTypes, password: bytes, encryption_algorithm: KeySerializationEncryption) -> bytes:
    """Serialize private key with OpenSSH custom encoding."""
    utils._check_bytes('password', password)
    if isinstance(private_key, dsa.DSAPrivateKey):
        warnings.warn('SSH DSA key support is deprecated and will be removed in a future release', utils.DeprecatedIn40, stacklevel=4)
    key_type = _get_ssh_key_type(private_key)
    kformat = _lookup_kformat(key_type)
    f_kdfoptions = _FragList()
    if password:
        ciphername = _DEFAULT_CIPHER
        blklen = _SSH_CIPHERS[ciphername].block_len
        kdfname = _BCRYPT
        rounds = _DEFAULT_ROUNDS
        if isinstance(encryption_algorithm, _KeySerializationEncryption) and encryption_algorithm._kdf_rounds is not None:
            rounds = encryption_algorithm._kdf_rounds
        salt = os.urandom(16)
        f_kdfoptions.put_sshstr(salt)
        f_kdfoptions.put_u32(rounds)
        ciph = _init_cipher(ciphername, password, salt, rounds)
    else:
        ciphername = kdfname = _NONE
        blklen = 8
        ciph = None
    nkeys = 1
    checkval = os.urandom(4)
    comment = b''
    f_public_key = _FragList()
    f_public_key.put_sshstr(key_type)
    kformat.encode_public(private_key.public_key(), f_public_key)
    f_secrets = _FragList([checkval, checkval])
    f_secrets.put_sshstr(key_type)
    kformat.encode_private(private_key, f_secrets)
    f_secrets.put_sshstr(comment)
    f_secrets.put_raw(_PADDING[:blklen - f_secrets.size() % blklen])
    f_main = _FragList()
    f_main.put_raw(_SK_MAGIC)
    f_main.put_sshstr(ciphername)
    f_main.put_sshstr(kdfname)
    f_main.put_sshstr(f_kdfoptions)
    f_main.put_u32(nkeys)
    f_main.put_sshstr(f_public_key)
    f_main.put_sshstr(f_secrets)
    slen = f_secrets.size()
    mlen = f_main.size()
    buf = memoryview(bytearray(mlen + blklen))
    f_main.render(buf)
    ofs = mlen - slen
    if ciph is not None:
        ciph.encryptor().update_into(buf[ofs:mlen], buf[ofs:])
    return _ssh_pem_encode(buf[:mlen])