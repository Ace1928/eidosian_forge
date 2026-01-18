from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
import os
import sys
import warnings
from passlib import exc, hash
from passlib.utils import repeat_string
from passlib.utils.compat import irange, PY3, u, get_method_function
from passlib.tests.utils import TestCase, HandlerCase, skipUnless, \
class bsd_nthash_test(HandlerCase):
    handler = hash.bsd_nthash
    known_correct_hashes = [('passphrase', '$3$$7f8fe03093cc84b267b109625f6bbf4b'), (b'\xc3\xbc', '$3$$8bd6e4fb88e01009818749c5443ea712')]
    known_unidentified_hashes = ['$3$$7f8fe03093cc84b267b109625f6bbfxb']