from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
import os
import sys
import warnings
from passlib import exc, hash
from passlib.utils import repeat_string
from passlib.utils.compat import irange, PY3, u, get_method_function
from passlib.tests.utils import TestCase, HandlerCase, skipUnless, \
class mysql323_test(HandlerCase):
    handler = hash.mysql323
    known_correct_hashes = [('drew', '697a7de87c5390b2'), ('password', '5d2e19393cc5ef67'), ('mypass', '6f8c114b58f2ce9e'), (UPASS_TABLE, '4ef327ca5491c8d7')]
    known_unidentified_hashes = ['6z8c114b58f2ce9e']

    def test_90_whitespace(self):
        """check whitespace is ignored per spec"""
        h = self.do_encrypt('mypass')
        h2 = self.do_encrypt('my pass')
        self.assertEqual(h, h2)

    class FuzzHashGenerator(HandlerCase.FuzzHashGenerator):

        def accept_password_pair(self, secret, other):
            return secret.replace(' ', '') != other.replace(' ', '')