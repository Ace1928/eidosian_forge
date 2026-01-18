from binascii import hexlify
import logging; log = logging.getLogger(__name__)
from warnings import warn
from passlib.utils import to_unicode, right_pad_string
from passlib.utils.compat import unicode
from passlib.crypto.digest import lookup_hash
import passlib.utils.handlers as uh
class msdcc2(uh.HasUserContext, uh.StaticHandler):
    """This class implements version 2 of Microsoft's Domain Cached Credentials
    password hash, and follows the :ref:`password-hash-api`.

    It has a fixed number of rounds, and uses the associated
    username as the salt.

    The :meth:`~passlib.ifc.PasswordHash.hash`, :meth:`~passlib.ifc.PasswordHash.genhash`, and :meth:`~passlib.ifc.PasswordHash.verify` methods
    have the following extra keyword:

    :type user: str
    :param user:
        String containing name of user account this password is associated with.
        This is required to properly calculate the hash.

        This keyword is case-insensitive, and should contain just the username
        (e.g. ``Administrator``, not ``SOMEDOMAIN\\Administrator``).
    """
    name = 'msdcc2'
    checksum_chars = uh.HEX_CHARS
    checksum_size = 32

    @classmethod
    def _norm_hash(cls, hash):
        return hash.lower()

    def _calc_checksum(self, secret):
        return hexlify(self.raw(secret, self.user)).decode('ascii')

    @classmethod
    def raw(cls, secret, user):
        """encode password using msdcc v2 algorithm

        :type secret: unicode or utf-8 bytes
        :arg secret: secret

        :type user: str
        :arg user: username to use as salt

        :returns: returns string of raw bytes
        """
        from passlib.crypto.digest import pbkdf2_hmac
        secret = to_unicode(secret, 'utf-8', param='secret').encode('utf-16-le')
        user = to_unicode(user, 'utf-8', param='user').lower().encode('utf-16-le')
        tmp = md4(md4(secret).digest() + user).digest()
        return pbkdf2_hmac('sha1', tmp, user, 10240, 16)