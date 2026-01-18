from __future__ import print_function
import re
import struct
import binascii
from collections import namedtuple
from Cryptodome.Util.py3compat import bord, tobytes, tostr, bchr, is_string
from Cryptodome.Util.number import bytes_to_long, long_to_bytes
from Cryptodome.Math.Numbers import Integer
from Cryptodome.Util.asn1 import (DerObjectId, DerOctetString, DerSequence,
from Cryptodome.Util._raw_api import (load_pycryptodome_raw_lib, VoidPointer,
from Cryptodome.PublicKey import (_expand_subject_public_key_info,
from Cryptodome.Hash import SHA512, SHAKE256
from Cryptodome.Random import get_random_bytes
from Cryptodome.Random.random import getrandbits
Export this ECC key.

        Args:
          format (string):
            The output format:

            - ``'DER'``. The key will be encoded in ASN.1 DER format (binary).
              For a public key, the ASN.1 ``subjectPublicKeyInfo`` structure
              defined in `RFC5480`_ will be used.
              For a private key, the ASN.1 ``ECPrivateKey`` structure defined
              in `RFC5915`_ is used instead (possibly within a PKCS#8 envelope,
              see the ``use_pkcs8`` flag below).
            - ``'PEM'``. The key will be encoded in a PEM_ envelope (ASCII).
            - ``'OpenSSH'``. The key will be encoded in the OpenSSH_ format
              (ASCII, public keys only).
            - ``'SEC1'``. The public key (i.e., the EC point) will be encoded
              into ``bytes`` according to Section 2.3.3 of `SEC1`_
              (which is a subset of the older X9.62 ITU standard).
              Only for NIST P-curves.
            - ``'raw'``. The public key will be encoded as ``bytes``,
              without any metadata.

              * For NIST P-curves: equivalent to ``'SEC1'``.
              * For EdDSA curves: ``bytes`` in the format defined in `RFC8032`_.

          passphrase (bytes or string):
            (*Private keys only*) The passphrase to protect the
            private key.

          use_pkcs8 (boolean):
            (*Private keys only*)
            If ``True`` (default and recommended), the `PKCS#8`_ representation
            will be used. It must be ``True`` for EdDSA curves.

            If ``False`` and a passphrase is present, the obsolete PEM
            encryption will be used.

          protection (string):
            When a private key is exported with password-protection
            and PKCS#8 (both ``DER`` and ``PEM`` formats), this parameter MUST be
            present,
            For all possible protection schemes,
            refer to :ref:`the encryption parameters of PKCS#8<enc_params>`.
            It is recommended to use ``'PBKDF2WithHMAC-SHA5126AndAES128-CBC'``.

          compress (boolean):
            If ``True``, the method returns a more compact representation
            of the public key, with the X-coordinate only.

            If ``False`` (default), the method returns the full public key.

            This parameter is ignored for EdDSA curves, as compression is
            mandatory.

          prot_params (dict):
            When a private key is exported with password-protection
            and PKCS#8 (both ``DER`` and ``PEM`` formats), this dictionary
            contains the  parameters to use to derive the encryption key
            from the passphrase.
            For all possible values,
            refer to :ref:`the encryption parameters of PKCS#8<enc_params>`.
            The recommendation is to use ``{'iteration_count':21000}`` for PBKDF2,
            and ``{'iteration_count':131072}`` for scrypt.

        .. warning::
            If you don't provide a passphrase, the private key will be
            exported in the clear!

        .. note::
            When exporting a private key with password-protection and `PKCS#8`_
            (both ``DER`` and ``PEM`` formats), any extra parameters
            to ``export_key()`` will be passed to :mod:`Cryptodome.IO.PKCS8`.

        .. _PEM:        http://www.ietf.org/rfc/rfc1421.txt
        .. _`PEM encryption`: http://www.ietf.org/rfc/rfc1423.txt
        .. _OpenSSH:    http://www.openssh.com/txt/rfc5656.txt
        .. _RFC5480:    https://tools.ietf.org/html/rfc5480
        .. _SEC1:       https://www.secg.org/sec1-v2.pdf

        Returns:
            A multi-line string (for ``'PEM'`` and ``'OpenSSH'``) or
            ``bytes`` (for ``'DER'``, ``'SEC1'``, and ``'raw'``) with the encoded key.
        