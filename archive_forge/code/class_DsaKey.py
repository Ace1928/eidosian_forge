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
class DsaKey(object):
    """Class defining an actual DSA key.
    Do not instantiate directly.
    Use :func:`generate`, :func:`construct` or :func:`import_key` instead.

    :ivar p: DSA modulus
    :vartype p: integer

    :ivar q: Order of the subgroup
    :vartype q: integer

    :ivar g: Generator
    :vartype g: integer

    :ivar y: Public key
    :vartype y: integer

    :ivar x: Private key
    :vartype x: integer

    :undocumented: exportKey, publickey
    """
    _keydata = ['y', 'g', 'p', 'q', 'x']

    def __init__(self, key_dict):
        input_set = set(key_dict.keys())
        public_set = set(('y', 'g', 'p', 'q'))
        if not public_set.issubset(input_set):
            raise ValueError('Some DSA components are missing = %s' % str(public_set - input_set))
        extra_set = input_set - public_set
        if extra_set and extra_set != set(('x',)):
            raise ValueError('Unknown DSA components = %s' % str(extra_set - set(('x',))))
        self._key = dict(key_dict)

    def _sign(self, m, k):
        if not self.has_private():
            raise TypeError('DSA public key cannot be used for signing')
        if not 1 < k < self.q:
            raise ValueError('k is not between 2 and q-1')
        x, q, p, g = [self._key[comp] for comp in ['x', 'q', 'p', 'g']]
        blind_factor = Integer.random_range(min_inclusive=1, max_exclusive=q)
        inv_blind_k = (blind_factor * k).inverse(q)
        blind_x = x * blind_factor
        r = pow(g, k, p) % q
        s = inv_blind_k * (blind_factor * m + blind_x * r) % q
        return map(int, (r, s))

    def _verify(self, m, sig):
        r, s = sig
        y, q, p, g = [self._key[comp] for comp in ['y', 'q', 'p', 'g']]
        if not 0 < r < q or not 0 < s < q:
            return False
        w = Integer(s).inverse(q)
        u1 = w * m % q
        u2 = w * r % q
        v = pow(g, u1, p) * pow(y, u2, p) % p % q
        return v == r

    def has_private(self):
        """Whether this is a DSA private key"""
        return 'x' in self._key

    def can_encrypt(self):
        return False

    def can_sign(self):
        return True

    def public_key(self):
        """A matching DSA public key.

        Returns:
            a new :class:`DsaKey` object
        """
        public_components = dict(((k, self._key[k]) for k in ('y', 'g', 'p', 'q')))
        return DsaKey(public_components)

    def __eq__(self, other):
        if bool(self.has_private()) != bool(other.has_private()):
            return False
        result = True
        for comp in self._keydata:
            result = result and getattr(self._key, comp, None) == getattr(other._key, comp, None)
        return result

    def __ne__(self, other):
        return not self.__eq__(other)

    def __getstate__(self):
        from pickle import PicklingError
        raise PicklingError

    def domain(self):
        """The DSA domain parameters.

        Returns
            tuple : (p,q,g)
        """
        return [int(self._key[comp]) for comp in ('p', 'q', 'g')]

    def __repr__(self):
        attrs = []
        for k in self._keydata:
            if k == 'p':
                bits = Integer(self.p).size_in_bits()
                attrs.append('p(%d)' % (bits,))
            elif hasattr(self, k):
                attrs.append(k)
        if self.has_private():
            attrs.append('private')
        return '<%s @0x%x %s>' % (self.__class__.__name__, id(self), ','.join(attrs))

    def __getattr__(self, item):
        try:
            return int(self._key[item])
        except KeyError:
            raise AttributeError(item)

    def export_key(self, format='PEM', pkcs8=None, passphrase=None, protection=None, randfunc=None):
        """Export this DSA key.

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
        """
        if passphrase is not None:
            passphrase = tobytes(passphrase)
        if randfunc is None:
            randfunc = Random.get_random_bytes
        if format == 'OpenSSH':
            tup1 = [self._key[x].to_bytes() for x in ('p', 'q', 'g', 'y')]

            def func(x):
                if bord(x[0]) & 128:
                    return bchr(0) + x
                else:
                    return x
            tup2 = [func(x) for x in tup1]
            keyparts = [b'ssh-dss'] + tup2
            keystring = b''.join([struct.pack('>I', len(kp)) + kp for kp in keyparts])
            return b'ssh-dss ' + binascii.b2a_base64(keystring)[:-1]
        params = DerSequence([self.p, self.q, self.g])
        if self.has_private():
            if pkcs8 is None:
                pkcs8 = True
            if pkcs8:
                if not protection:
                    protection = 'PBKDF2WithHMAC-SHA1AndDES-EDE3-CBC'
                private_key = DerInteger(self.x).encode()
                binary_key = PKCS8.wrap(private_key, oid, passphrase, protection, key_params=params, randfunc=randfunc)
                if passphrase:
                    key_type = 'ENCRYPTED PRIVATE'
                else:
                    key_type = 'PRIVATE'
                passphrase = None
            else:
                if format != 'PEM' and passphrase:
                    raise ValueError('DSA private key cannot be encrypted')
                ints = [0, self.p, self.q, self.g, self.y, self.x]
                binary_key = DerSequence(ints).encode()
                key_type = 'DSA PRIVATE'
        else:
            if pkcs8:
                raise ValueError('PKCS#8 is only meaningful for private keys')
            binary_key = _create_subject_public_key_info(oid, DerInteger(self.y), params)
            key_type = 'PUBLIC'
        if format == 'DER':
            return binary_key
        if format == 'PEM':
            pem_str = PEM.encode(binary_key, key_type + ' KEY', passphrase, randfunc)
            return tobytes(pem_str)
        raise ValueError("Unknown key format '%s'. Cannot export the DSA key." % format)
    exportKey = export_key
    publickey = public_key

    def sign(self, M, K):
        raise NotImplementedError('Use module Cryptodome.Signature.DSS instead')

    def verify(self, M, signature):
        raise NotImplementedError('Use module Cryptodome.Signature.DSS instead')

    def encrypt(self, plaintext, K):
        raise NotImplementedError

    def decrypt(self, ciphertext):
        raise NotImplementedError

    def blind(self, M, B):
        raise NotImplementedError

    def unblind(self, M, B):
        raise NotImplementedError

    def size(self):
        raise NotImplementedError