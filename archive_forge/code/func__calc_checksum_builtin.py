from hashlib import md5
import logging; log = logging.getLogger(__name__)
from passlib.utils import safe_crypt, test_crypt, repeat_string
from passlib.utils.binary import h64
from passlib.utils.compat import unicode, u
import passlib.utils.handlers as uh
def _calc_checksum_builtin(self, secret):
    return _raw_md5_crypt(secret, self.salt)