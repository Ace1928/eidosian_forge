import binascii
import struct
import itertools
from Cryptodome.Util.py3compat import bchr, bord, tobytes, tostr, iter_range
from Cryptodome import Random
from Cryptodome.IO import PKCS8, PEM
from Cryptodome.Hash import SHA256
from Cryptodome.Util.asn1 import (
from Cryptodome.Math.Numbers import Integer
from Cryptodome.Math.Primality import (test_probable_prime, COMPOSITE,
from Cryptodome.PublicKey import (_expand_subject_public_key_info,
importKey = import_key
Export this DSA key.

        Args:
          format (string):
            The encoding for the output:

            - *'PEM'* (default). ASCII as per `RFC1421`_/ `RFC1423`_.
            - *'DER'*. Binary ASN.1 encoding.
            - *'OpenSSH'*. ASCII one-liner as per `RFC4253`_.
              Only suitable for public keys, not for private keys.

          passphrase (string):
            *Private keys only*. The pass phrase to protect the output.

          pkcs8 (boolean):
            *Private keys only*. If ``True`` (default), the key is encoded
            with `PKCS#8`_. If ``False``, it is encoded in the custom
            OpenSSL/OpenSSH container.

          protection (string):
            *Only in combination with a pass phrase*.
            The encryption scheme to use to protect the output.

            If :data:`pkcs8` takes value ``True``, this is the PKCS#8
            algorithm to use for deriving the secret and encrypting
            the private DSA key.
            For a complete list of algorithms, see :mod:`Cryptodome.IO.PKCS8`.
            The default is *PBKDF2WithHMAC-SHA1AndDES-EDE3-CBC*.

            If :data:`pkcs8` is ``False``, the obsolete PEM encryption scheme is
            used. It is based on MD5 for key derivation, and Triple DES for
            encryption. Parameter :data:`protection` is then ignored.

            The combination ``format='DER'`` and ``pkcs8=False`` is not allowed
            if a passphrase is present.

          randfunc (callable):
            A function that returns random bytes.
            By default it is :func:`Cryptodome.Random.get_random_bytes`.

        Returns:
          byte string : the encoded key

        Raises:
          ValueError : when the format is unknown or when you try to encrypt a private
            key with *DER* format and OpenSSL/OpenSSH.

        .. warning::
            If you don't provide a pass phrase, the private key will be
            exported in the clear!

        .. _RFC1421:    http://www.ietf.org/rfc/rfc1421.txt
        .. _RFC1423:    http://www.ietf.org/rfc/rfc1423.txt
        .. _RFC4253:    http://www.ietf.org/rfc/rfc4253.txt
        .. _`PKCS#8`:   http://www.ietf.org/rfc/rfc5208.txt
        