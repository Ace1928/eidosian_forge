from base64 import b64encode, b64decode
from hashlib import md5, sha1, sha256, sha512
import logging; log = logging.getLogger(__name__)
import re
from passlib.handlers.misc import plaintext
from passlib.utils import unix_crypt_schemes, to_unicode
from passlib.utils.compat import uascii_to_str, unicode, u
from passlib.utils.decor import classproperty
import passlib.utils.handlers as uh
class ldap_plaintext(plaintext):
    """This class stores passwords in plaintext, and follows the :ref:`password-hash-api`.

    This class acts much like the generic :class:`!passlib.hash.plaintext` handler,
    except that it will identify a hash only if it does NOT begin with the ``{XXX}`` identifier prefix
    used by RFC2307 passwords.

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
    name = 'ldap_plaintext'
    _2307_pat = re.compile(u('^\\{\\w+\\}.*$'))

    @uh.deprecated_method(deprecated='1.7', removed='2.0')
    @classmethod
    def genconfig(cls):
        return '!'

    @classmethod
    def identify(cls, hash):
        hash = uh.to_unicode_for_identify(hash)
        return bool(hash) and cls._2307_pat.match(hash) is None