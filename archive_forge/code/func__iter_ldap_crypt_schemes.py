import logging; log = logging.getLogger(__name__)
from itertools import chain
from passlib import hash
from passlib.context import LazyCryptContext
from passlib.utils import sys_bits
def _iter_ldap_crypt_schemes():
    from passlib.utils import unix_crypt_schemes
    return ('ldap_' + name for name in unix_crypt_schemes)