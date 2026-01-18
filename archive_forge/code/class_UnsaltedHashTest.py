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
class UnsaltedHashTest(HandlerCase):
    handler = UnsaltedHash
    known_correct_hashes = [('password', '61cfd32684c47de231f1f982c214e884133762c0'), (UPASS_TEMP, '96b329d120b97ff81ada770042e44ba87343ad2b')]

    def test_bad_kwds(self):
        self.assertRaises(TypeError, UnsaltedHash, salt='x')
        self.assertRaises(TypeError, UnsaltedHash.genconfig, rounds=1)