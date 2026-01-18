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
class SaltedHashTest(HandlerCase):
    handler = SaltedHash
    known_correct_hashes = [('password', '@salt77d71f8fe74f314dac946766c1ac4a2a58365482c0'), (UPASS_TEMP, '@salt9f978a9bfe360d069b0c13f2afecd570447407fa7e48')]

    def test_bad_kwds(self):
        stub = SaltedHash(use_defaults=True)._stub_checksum
        self.assertRaises(TypeError, SaltedHash, checksum=stub, salt=None)
        self.assertRaises(ValueError, SaltedHash, checksum=stub, salt='xxx')