import binascii
import struct
from Cryptodome import Random
from Cryptodome.Util.py3compat import tobytes, bord, tostr
from Cryptodome.Util.asn1 import DerSequence, DerNull
from Cryptodome.Util.number import bytes_to_long
from Cryptodome.Math.Numbers import Integer
from Cryptodome.Math.Primality import (test_probable_prime,
from Cryptodome.PublicKey import (_expand_subject_public_key_info,
importKey = import_key
def _import_openssh_private_rsa(data, password):
    from ._openssh import import_openssh_private_generic, read_bytes, read_string, check_padding
    ssh_name, decrypted = import_openssh_private_generic(data, password)
    if ssh_name != 'ssh-rsa':
        raise ValueError('This SSH key is not RSA')
    n, decrypted = read_bytes(decrypted)
    e, decrypted = read_bytes(decrypted)
    d, decrypted = read_bytes(decrypted)
    iqmp, decrypted = read_bytes(decrypted)
    p, decrypted = read_bytes(decrypted)
    q, decrypted = read_bytes(decrypted)
    _, padded = read_string(decrypted)
    check_padding(padded)
    build = [Integer.from_bytes(x) for x in (n, e, d, q, p, iqmp)]
    return construct(build)