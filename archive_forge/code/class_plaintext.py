import sys
import logging; log = logging.getLogger(__name__)
from warnings import warn
from passlib.utils import to_native_str, str_consteq
from passlib.utils.compat import unicode, u, unicode_or_bytes_types
import passlib.utils.handlers as uh
class plaintext(uh.MinimalHandler):
    """This class stores passwords in plaintext, and follows the :ref:`password-hash-api`.

    The :meth:`~passlib.ifc.PasswordHash.hash`, :meth:`~passlib.ifc.PasswordHash.genhash`, and :meth:`~passlib.ifc.PasswordHash.verify` methods all require the
    following additional contextual keyword:

    :type encoding: str
    :param encoding:
        This controls the character encoding to use (defaults to ``utf-8``).

        This encoding will be used to encode :class:`!unicode` passwords
        under Python 2, and decode :class:`!bytes` hashes under Python 3.

    .. versionchanged:: 1.6
        The ``encoding`` keyword was added.
    """
    name = 'plaintext'
    setting_kwds = ()
    context_kwds = ('encoding',)
    default_encoding = 'utf-8'

    @classmethod
    def identify(cls, hash):
        if isinstance(hash, unicode_or_bytes_types):
            return True
        else:
            raise uh.exc.ExpectedStringError(hash, 'hash')

    @classmethod
    def hash(cls, secret, encoding=None):
        uh.validate_secret(secret)
        if not encoding:
            encoding = cls.default_encoding
        return to_native_str(secret, encoding, 'secret')

    @classmethod
    def verify(cls, secret, hash, encoding=None):
        if not encoding:
            encoding = cls.default_encoding
        hash = to_native_str(hash, encoding, 'hash')
        if not cls.identify(hash):
            raise uh.exc.InvalidHashError(cls)
        return str_consteq(cls.hash(secret, encoding), hash)

    @uh.deprecated_method(deprecated='1.7', removed='2.0')
    @classmethod
    def genconfig(cls):
        return cls.hash('')

    @uh.deprecated_method(deprecated='1.7', removed='2.0')
    @classmethod
    def genhash(cls, secret, config, encoding=None):
        if not cls.identify(config):
            raise uh.exc.InvalidHashError(cls)
        return cls.hash(secret, encoding=encoding)