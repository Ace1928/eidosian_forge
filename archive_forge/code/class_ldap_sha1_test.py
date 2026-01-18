from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
import os
import sys
import warnings
from passlib import exc, hash
from passlib.utils import repeat_string
from passlib.utils.compat import irange, PY3, u, get_method_function
from passlib.tests.utils import TestCase, HandlerCase, skipUnless, \
class ldap_sha1_test(HandlerCase):
    handler = hash.ldap_sha1
    known_correct_hashes = [('helloworld', '{SHA}at+xg6SiyUovktq1redipHiJpaE='), (UPASS_TABLE, '{SHA}4FmyYo46Pi3glWed6YIsHRRm4PA=')]