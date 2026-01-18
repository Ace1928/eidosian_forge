from base64 import b64encode, b64decode
from hashlib import md5, sha1, sha256, sha512
import logging; log = logging.getLogger(__name__)
import re
from passlib.handlers.misc import plaintext
from passlib.utils import unix_crypt_schemes, to_unicode
from passlib.utils.compat import uascii_to_str, unicode, u
from passlib.utils.decor import classproperty
import passlib.utils.handlers as uh
def _init_ldap_crypt_handlers():
    g = globals()
    for wname in unix_crypt_schemes:
        name = 'ldap_' + wname
        g[name] = uh.PrefixWrapper(name, wname, prefix=u('{CRYPT}'), lazy=True)
    del g