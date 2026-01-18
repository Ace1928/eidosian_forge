from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
import os
import sys
import warnings
from passlib import exc, hash
from passlib.utils import repeat_string
from passlib.utils.compat import irange, PY3, u, get_method_function
from passlib.tests.utils import TestCase, HandlerCase, skipUnless, \
class ldap_plaintext_test(HandlerCase):
    handler = hash.ldap_plaintext
    known_correct_hashes = [('password', 'password'), (UPASS_TABLE, UPASS_TABLE if PY3 else PASS_TABLE_UTF8), (PASS_TABLE_UTF8, UPASS_TABLE if PY3 else PASS_TABLE_UTF8)]
    known_unidentified_hashes = ['{FOO}bar', '']
    known_other_hashes = [('ldap_md5', '{MD5}/F4DjTilcDIIVEHn/nAQsA==')]

    class FuzzHashGenerator(HandlerCase.FuzzHashGenerator):

        def random_password(self):
            while True:
                pwd = super(ldap_plaintext_test.FuzzHashGenerator, self).random_password()
                if pwd:
                    return pwd