import itertools
from oslo_log import log
import passlib.hash
import keystone.conf
from keystone import exception
from keystone.i18n import _
def _get_hasher_from_ident(hashed):
    try:
        return _HASHER_IDENT_MAP[hashed[0:hashed.index('$', 1) + 1]]
    except KeyError:
        raise ValueError(_('Unsupported password hashing algorithm ident: %s') % hashed[0:hashed.index('$', 1) + 1])