from hashlib import md5
import logging; log = logging.getLogger(__name__)
from passlib.utils import safe_crypt, test_crypt, repeat_string
from passlib.utils.binary import h64
from passlib.utils.compat import unicode, u
import passlib.utils.handlers as uh
class _MD5_Common(uh.HasSalt, uh.GenericHandler):
    """common code for md5_crypt and apr_md5_crypt"""
    setting_kwds = ('salt', 'salt_size')
    checksum_size = 22
    checksum_chars = uh.HASH64_CHARS
    max_salt_size = 8
    salt_chars = uh.HASH64_CHARS

    @classmethod
    def from_string(cls, hash):
        salt, chk = uh.parse_mc2(hash, cls.ident, handler=cls)
        return cls(salt=salt, checksum=chk)

    def to_string(self):
        return uh.render_mc2(self.ident, self.salt, self.checksum)