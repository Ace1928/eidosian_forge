from __future__ import with_statement
import re
import hashlib
from logging import getLogger
import warnings
from passlib.hash import ldap_md5, sha256_crypt
from passlib.exc import MissingBackendError, PasslibHashWarning
from passlib.utils.compat import str_to_uascii, \
import passlib.utils.handlers as uh
from passlib.tests.utils import HandlerCase, TestCase
from passlib.utils.compat import u
class d1(uh.HasManyIdents, uh.GenericHandler):
    name = 'd1'
    setting_kwds = ('ident',)
    default_ident = u('!A')
    ident_values = (u('!A'), u('!B'))
    ident_aliases = {u('A'): u('!A')}