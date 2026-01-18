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
class EccKey(object):
    """Class defining an ECC key.
    Do not instantiate directly.
    Use :func:`generate`, :func:`construct` or :func:`import_key` instead.

    :ivar curve: The name of the curve as defined in the `ECC table`_.
    :vartype curve: string

    :ivar pointQ: an ECC point representating the public component.
    :vartype pointQ: :class:`EccPoint`

    :ivar d: A scalar that represents the private component
             in NIST P curves. It is smaller than the
             order of the generator point.
    :vartype d: integer

    :ivar seed: A seed that representats the private component
                in EdDSA curves
                (Ed25519, 32 bytes; Ed448, 57 bytes).
    :vartype seed: bytes
    """

    def __init__(self, **kwargs):
        """Create a new ECC key

        Keywords:
          curve : string
            The name of the curve.
          d : integer
            Mandatory for a private key one NIST P curves.
            It must be in the range ``[1..order-1]``.
          seed : bytes
            Mandatory for a private key on the Ed25519 (32 bytes)
            or Ed448 (57 bytes) curve.
          point : EccPoint
            Mandatory for a public key. If provided for a private key,
            the implementation will NOT check whether it matches ``d``.

        Only one parameter among ``d``, ``seed`` or ``point`` may be used.
        """
        kwargs_ = dict(kwargs)
        curve_name = kwargs_.pop('curve', None)
        self._d = kwargs_.pop('d', None)
        self._seed = kwargs_.pop('seed', None)
        self._point = kwargs_.pop('point', None)
        if curve_name is None and self._point:
            curve_name = self._point._curve_name
        if kwargs_:
            raise TypeError('Unknown parameters: ' + str(kwargs_))
        if curve_name not in _curves:
            raise ValueError('Unsupported curve (%s)' % curve_name)
        self._curve = _curves[curve_name]
        self.curve = self._curve.desc
        count = int(self._d is not None) + int(self._seed is not None)
        if count == 0:
            if self._point is None:
                raise ValueError("At lest one between parameters 'point', 'd' or 'seed' must be specified")
            return
        if count == 2:
            raise ValueError('Parameters d and seed are mutually exclusive')
        if not self._is_eddsa():
            if self._seed is not None:
                raise ValueError("Parameter 'seed' can only be used with Ed25519 or Ed448")
            self._d = Integer(self._d)
            if not 1 <= self._d < self._curve.order:
                raise ValueError('Parameter d must be an integer smaller than the curve order')
        else:
            if self._d is not None:
                raise ValueError('Parameter d can only be used with NIST P curves')
            if self._curve.name == 'ed25519':
                if len(self._seed) != 32:
                    raise ValueError('Parameter seed must be 32 bytes long for Ed25519')
                seed_hash = SHA512.new(self._seed).digest()
                self._prefix = seed_hash[32:]
                tmp = bytearray(seed_hash[:32])
                tmp[0] &= 248
                tmp[31] = tmp[31] & 127 | 64
            elif self._curve.name == 'ed448':
                if len(self._seed) != 57:
                    raise ValueError('Parameter seed must be 57 bytes long for Ed448')
                seed_hash = SHAKE256.new(self._seed).read(114)
                self._prefix = seed_hash[57:]
                tmp = bytearray(seed_hash[:57])
                tmp[0] &= 252
                tmp[55] |= 128
                tmp[56] = 0
            self._d = Integer.from_bytes(tmp, byteorder='little')

    def _is_eddsa(self):
        return self._curve.desc in ('Ed25519', 'Ed448')

    def __eq__(self, other):
        if not isinstance(other, EccKey):
            return False
        if other.has_private() != self.has_private():
            return False
        return other.pointQ == self.pointQ

    def __repr__(self):
        if self.has_private():
            if self._is_eddsa():
                extra = ', seed=%s' % tostr(binascii.hexlify(self._seed))
            else:
                extra = ', d=%d' % int(self._d)
        else:
            extra = ''
        x, y = self.pointQ.xy
        return "EccKey(curve='%s', point_x=%d, point_y=%d%s)" % (self._curve.desc, x, y, extra)

    def has_private(self):
        """``True`` if this key can be used for making signatures or decrypting data."""
        return self._d is not None

    def _sign(self, z, k):
        assert 0 < k < self._curve.order
        order = self._curve.order
        blind = Integer.random_range(min_inclusive=1, max_exclusive=order)
        blind_d = self._d * blind
        inv_blind_k = (blind * k).inverse(order)
        r = (self._curve.G * k).x % order
        s = inv_blind_k * (blind * z + blind_d * r) % order
        return (r, s)

    def _verify(self, z, rs):
        order = self._curve.order
        sinv = rs[1].inverse(order)
        point1 = self._curve.G * (sinv * z % order)
        point2 = self.pointQ * (sinv * rs[0] % order)
        return (point1 + point2).x == rs[0]

    @property
    def d(self):
        if not self.has_private():
            raise ValueError('This is not a private ECC key')
        return self._d

    @property
    def seed(self):
        if not self.has_private():
            raise ValueError('This is not a private ECC key')
        return self._seed

    @property
    def pointQ(self):
        if self._point is None:
            self._point = self._curve.G * self._d
        return self._point

    def public_key(self):
        """A matching ECC public key.

        Returns:
            a new :class:`EccKey` object
        """
        return EccKey(curve=self._curve.desc, point=self.pointQ)

    def _export_SEC1(self, compress):
        if self._is_eddsa():
            raise ValueError('SEC1 format is unsupported for EdDSA curves')
        modulus_bytes = self.pointQ.size_in_bytes()
        if compress:
            if self.pointQ.y.is_odd():
                first_byte = b'\x03'
            else:
                first_byte = b'\x02'
            public_key = first_byte + self.pointQ.x.to_bytes(modulus_bytes)
        else:
            public_key = b'\x04' + self.pointQ.x.to_bytes(modulus_bytes) + self.pointQ.y.to_bytes(modulus_bytes)
        return public_key

    def _export_eddsa(self):
        x, y = self.pointQ.xy
        if self._curve.name == 'ed25519':
            result = bytearray(y.to_bytes(32, byteorder='little'))
            result[31] = (x & 1) << 7 | result[31]
        elif self._curve.name == 'ed448':
            result = bytearray(y.to_bytes(57, byteorder='little'))
            result[56] = (x & 1) << 7
        else:
            raise ValueError('Not an EdDSA key to export')
        return bytes(result)

    def _export_subjectPublicKeyInfo(self, compress):
        if self._is_eddsa():
            oid = self._curve.oid
            public_key = self._export_eddsa()
            params = None
        else:
            oid = '1.2.840.10045.2.1'
            public_key = self._export_SEC1(compress)
            params = DerObjectId(self._curve.oid)
        return _create_subject_public_key_info(oid, public_key, params)

    def _export_rfc5915_private_der(self, include_ec_params=True):
        assert self.has_private()
        modulus_bytes = self.pointQ.size_in_bytes()
        public_key = b'\x04' + self.pointQ.x.to_bytes(modulus_bytes) + self.pointQ.y.to_bytes(modulus_bytes)
        seq = [1, DerOctetString(self.d.to_bytes(modulus_bytes)), DerObjectId(self._curve.oid, explicit=0), DerBitString(public_key, explicit=1)]
        if not include_ec_params:
            del seq[2]
        return DerSequence(seq).encode()

    def _export_pkcs8(self, **kwargs):
        from Cryptodome.IO import PKCS8
        if kwargs.get('passphrase', None) is not None and 'protection' not in kwargs:
            raise ValueError("At least the 'protection' parameter must be present")
        if self._is_eddsa():
            oid = self._curve.oid
            private_key = DerOctetString(self._seed).encode()
            params = None
        else:
            oid = '1.2.840.10045.2.1'
            private_key = self._export_rfc5915_private_der(include_ec_params=False)
            params = DerObjectId(self._curve.oid)
        result = PKCS8.wrap(private_key, oid, key_params=params, **kwargs)
        return result

    def _export_public_pem(self, compress):
        from Cryptodome.IO import PEM
        encoded_der = self._export_subjectPublicKeyInfo(compress)
        return PEM.encode(encoded_der, 'PUBLIC KEY')

    def _export_private_pem(self, passphrase, **kwargs):
        from Cryptodome.IO import PEM
        encoded_der = self._export_rfc5915_private_der()
        return PEM.encode(encoded_der, 'EC PRIVATE KEY', passphrase, **kwargs)

    def _export_private_clear_pkcs8_in_clear_pem(self):
        from Cryptodome.IO import PEM
        encoded_der = self._export_pkcs8()
        return PEM.encode(encoded_der, 'PRIVATE KEY')

    def _export_private_encrypted_pkcs8_in_clear_pem(self, passphrase, **kwargs):
        from Cryptodome.IO import PEM
        assert passphrase
        if 'protection' not in kwargs:
            raise ValueError("At least the 'protection' parameter should be present")
        encoded_der = self._export_pkcs8(passphrase=passphrase, **kwargs)
        return PEM.encode(encoded_der, 'ENCRYPTED PRIVATE KEY')

    def _export_openssh(self, compress):
        if self.has_private():
            raise ValueError('Cannot export OpenSSH private keys')
        desc = self._curve.openssh
        if desc is None:
            raise ValueError('Cannot export %s keys as OpenSSH' % self._curve.name)
        elif desc == 'ssh-ed25519':
            public_key = self._export_eddsa()
            comps = (tobytes(desc), tobytes(public_key))
        else:
            modulus_bytes = self.pointQ.size_in_bytes()
            if compress:
                first_byte = 2 + self.pointQ.y.is_odd()
                public_key = bchr(first_byte) + self.pointQ.x.to_bytes(modulus_bytes)
            else:
                public_key = b'\x04' + self.pointQ.x.to_bytes(modulus_bytes) + self.pointQ.y.to_bytes(modulus_bytes)
            middle = desc.split('-')[2]
            comps = (tobytes(desc), tobytes(middle), public_key)
        blob = b''.join([struct.pack('>I', len(x)) + x for x in comps])
        return desc + ' ' + tostr(binascii.b2a_base64(blob))

    def export_key(self, **kwargs):
        """Export this ECC key.

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
        """
        args = kwargs.copy()
        ext_format = args.pop('format')
        if ext_format not in ('PEM', 'DER', 'OpenSSH', 'SEC1', 'raw'):
            raise ValueError("Unknown format '%s'" % ext_format)
        compress = args.pop('compress', False)
        if self.has_private():
            passphrase = args.pop('passphrase', None)
            if is_string(passphrase):
                passphrase = tobytes(passphrase)
                if not passphrase:
                    raise ValueError('Empty passphrase')
            use_pkcs8 = args.pop('use_pkcs8', True)
            if not use_pkcs8:
                if self._is_eddsa():
                    raise ValueError("'pkcs8' must be True for EdDSA curves")
                if 'protection' in args:
                    raise ValueError("'protection' is only supported for PKCS#8")
            if ext_format == 'PEM':
                if use_pkcs8:
                    if passphrase:
                        return self._export_private_encrypted_pkcs8_in_clear_pem(passphrase, **args)
                    else:
                        return self._export_private_clear_pkcs8_in_clear_pem()
                else:
                    return self._export_private_pem(passphrase, **args)
            elif ext_format == 'DER':
                if passphrase and (not use_pkcs8):
                    raise ValueError('Private keys can only be encrpyted with DER using PKCS#8')
                if use_pkcs8:
                    return self._export_pkcs8(passphrase=passphrase, **args)
                else:
                    return self._export_rfc5915_private_der()
            else:
                raise ValueError("Private keys cannot be exported in the '%s' format" % ext_format)
        else:
            if args:
                raise ValueError("Unexpected parameters: '%s'" % args)
            if ext_format == 'PEM':
                return self._export_public_pem(compress)
            elif ext_format == 'DER':
                return self._export_subjectPublicKeyInfo(compress)
            elif ext_format == 'SEC1':
                return self._export_SEC1(compress)
            elif ext_format == 'raw':
                if self._curve.name in ('ed25519', 'ed448'):
                    return self._export_eddsa()
                else:
                    return self._export_SEC1(compress)
            else:
                return self._export_openssh(compress)