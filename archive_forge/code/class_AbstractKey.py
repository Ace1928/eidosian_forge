import logging
import warnings
from rsa._compat import range
import rsa.prime
import rsa.pem
import rsa.common
import rsa.randnum
import rsa.core
class AbstractKey(object):
    """Abstract superclass for private and public keys."""
    __slots__ = ('n', 'e')

    def __init__(self, n, e):
        self.n = n
        self.e = e

    @classmethod
    def _load_pkcs1_pem(cls, keyfile):
        """Loads a key in PKCS#1 PEM format, implement in a subclass.

        :param keyfile: contents of a PEM-encoded file that contains
            the public key.
        :type keyfile: bytes

        :return: the loaded key
        :rtype: AbstractKey
        """

    @classmethod
    def _load_pkcs1_der(cls, keyfile):
        """Loads a key in PKCS#1 PEM format, implement in a subclass.

        :param keyfile: contents of a DER-encoded file that contains
            the public key.
        :type keyfile: bytes

        :return: the loaded key
        :rtype: AbstractKey
        """

    def _save_pkcs1_pem(self):
        """Saves the key in PKCS#1 PEM format, implement in a subclass.

        :returns: the PEM-encoded key.
        :rtype: bytes
        """

    def _save_pkcs1_der(self):
        """Saves the key in PKCS#1 DER format, implement in a subclass.

        :returns: the DER-encoded key.
        :rtype: bytes
        """

    @classmethod
    def load_pkcs1(cls, keyfile, format='PEM'):
        """Loads a key in PKCS#1 DER or PEM format.

        :param keyfile: contents of a DER- or PEM-encoded file that contains
            the key.
        :type keyfile: bytes
        :param format: the format of the file to load; 'PEM' or 'DER'
        :type format: str

        :return: the loaded key
        :rtype: AbstractKey
        """
        methods = {'PEM': cls._load_pkcs1_pem, 'DER': cls._load_pkcs1_der}
        method = cls._assert_format_exists(format, methods)
        return method(keyfile)

    @staticmethod
    def _assert_format_exists(file_format, methods):
        """Checks whether the given file format exists in 'methods'.
        """
        try:
            return methods[file_format]
        except KeyError:
            formats = ', '.join(sorted(methods.keys()))
            raise ValueError('Unsupported format: %r, try one of %s' % (file_format, formats))

    def save_pkcs1(self, format='PEM'):
        """Saves the key in PKCS#1 DER or PEM format.

        :param format: the format to save; 'PEM' or 'DER'
        :type format: str
        :returns: the DER- or PEM-encoded key.
        :rtype: bytes
        """
        methods = {'PEM': self._save_pkcs1_pem, 'DER': self._save_pkcs1_der}
        method = self._assert_format_exists(format, methods)
        return method()

    def blind(self, message, r):
        """Performs blinding on the message using random number 'r'.

        :param message: the message, as integer, to blind.
        :type message: int
        :param r: the random number to blind with.
        :type r: int
        :return: the blinded message.
        :rtype: int

        The blinding is such that message = unblind(decrypt(blind(encrypt(message))).

        See https://en.wikipedia.org/wiki/Blinding_%28cryptography%29
        """
        return message * pow(r, self.e, self.n) % self.n

    def unblind(self, blinded, r):
        """Performs blinding on the message using random number 'r'.

        :param blinded: the blinded message, as integer, to unblind.
        :param r: the random number to unblind with.
        :return: the original message.

        The blinding is such that message = unblind(decrypt(blind(encrypt(message))).

        See https://en.wikipedia.org/wiki/Blinding_%28cryptography%29
        """
        return rsa.common.inverse(r, self.n) * blinded % self.n