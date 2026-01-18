from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
import os
from warnings import warn
from passlib import exc, registry
from passlib.context import CryptContext
from passlib.exc import ExpectedStringError
from passlib.hash import htdigest
from passlib.utils import render_bytes, to_bytes, is_ascii_codec
from passlib.utils.decor import deprecated_method
from passlib.utils.compat import join_bytes, unicode, BytesIO, PY3
def _init_htpasswd_context():
    schemes = ['bcrypt', 'sha256_crypt', 'sha512_crypt', 'des_crypt', 'apr_md5_crypt', 'ldap_sha1', 'plaintext']
    schemes.extend(registry.get_supported_os_crypt_schemes())
    preferred = schemes[:3] + ['apr_md5_crypt'] + schemes
    schemes = sorted(set(schemes), key=preferred.index)
    return CryptContext(schemes=schemes, default=htpasswd_defaults['portable_apache_22'], bcrypt__ident='2y')