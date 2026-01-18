import logging; log = logging.getLogger(__name__)
from itertools import chain
from passlib import hash
from passlib.context import LazyCryptContext
from passlib.utils import sys_bits
def _load_master_config():
    from passlib.registry import list_crypt_handlers
    schemes = list_crypt_handlers()
    excluded = ['bigcrypt', 'crypt16', 'cisco_pix', 'cisco_type7', 'htdigest', 'mysql323', 'oracle10', 'lmhash', 'msdcc', 'msdcc2', 'nthash', 'plaintext', 'ldap_plaintext', 'django_disabled', 'unix_disabled', 'unix_fallback']
    for name in excluded:
        schemes.remove(name)
    return dict(schemes=schemes, default='sha256_crypt')