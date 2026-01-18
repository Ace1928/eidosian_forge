from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
import os
import sys
import warnings
from passlib import exc, hash
from passlib.utils import repeat_string
from passlib.utils.compat import irange, PY3, u, get_method_function
from passlib.tests.utils import TestCase, HandlerCase, skipUnless, \
class ldap_md5_test(HandlerCase):
    handler = hash.ldap_md5
    known_correct_hashes = [('helloworld', '{MD5}/F4DjTilcDIIVEHn/nAQsA=='), (UPASS_TABLE, '{MD5}BUc/ihn2aBXnN7MyZKDQsA==')]