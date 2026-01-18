from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
import os
import sys
import warnings
from passlib import exc, hash
from passlib.utils import repeat_string
from passlib.utils.compat import irange, PY3, u, get_method_function
from passlib.tests.utils import TestCase, HandlerCase, skipUnless, \
class apr_md5_crypt_test(HandlerCase):
    handler = hash.apr_md5_crypt
    known_correct_hashes = [('myPassword', '$apr1$r31.....$HqJZimcKQFAMYayBlzkrA/'), (UPASS_TABLE, '$apr1$bzYrOHUx$a1FcpXuQDJV3vPY20CS6N1')]
    known_malformed_hashes = ['$apr1$r31.....$HqJZimcKQFAMYayBlzkrA!']