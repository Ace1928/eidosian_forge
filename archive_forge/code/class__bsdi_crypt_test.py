from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
import os
import sys
import warnings
from passlib import exc, hash
from passlib.utils import repeat_string
from passlib.utils.compat import irange, PY3, u, get_method_function
from passlib.tests.utils import TestCase, HandlerCase, skipUnless, \
class _bsdi_crypt_test(HandlerCase):
    """test BSDiCrypt algorithm"""
    handler = hash.bsdi_crypt
    known_correct_hashes = [('U*U*U*U*', '_J9..CCCCXBrJUJV154M'), ('U*U***U', '_J9..CCCCXUhOBTXzaiE'), ('U*U***U*', '_J9..CCCC4gQ.mB/PffM'), ('*U*U*U*U', '_J9..XXXXvlzQGqpPPdk'), ('*U*U*U*U*', '_J9..XXXXsqM/YSSP..Y'), ('*U*U*U*U*U*U*U*U', '_J9..XXXXVL7qJCnku0I'), ('*U*U*U*U*U*U*U*U*', '_J9..XXXXAj8cFbP5scI'), ('ab1234567', '_J9..SDizh.vll5VED9g'), ('cr1234567', '_J9..SDizRjWQ/zePPHc'), ('zxyDPWgydbQjgq', '_J9..SDizxmRI1GjnQuE'), ('726 even', '_K9..SaltNrQgIYUAeoY'), ('', '_J9..SDSD5YGyRCr4W4c'), (' ', '_K1..crsmZxOLzfJH8iw'), ('my', '_KR/.crsmykRplHbAvwA'), ('my socra', '_K1..crsmf/9NzZr1fLM'), ('my socrates', '_K1..crsmOv1rbde9A9o'), ('my socrates note', '_K1..crsm/2qeAhdISMA'), (UPASS_TABLE, '_7C/.ABw0WIKy0ILVqo2')]
    known_unidentified_hashes = ['_K1.!crsmZxOLzfJH8iw']
    platform_crypt_support = [('openbsd[6789]', False), ('openbsd5', None), ('openbsd', True), ('freebsd|netbsd|darwin', True), ('solaris', False), ('linux', None)]

    def test_77_fuzz_input(self, **kwds):
        warnings.filterwarnings('ignore', 'bsdi_crypt rounds should be odd.*')
        super(_bsdi_crypt_test, self).test_77_fuzz_input(**kwds)

    def test_needs_update_w_even_rounds(self):
        """needs_update() should flag even rounds"""
        handler = self.handler
        even_hash = '_Y/../cG0zkJa6LY6k4c'
        odd_hash = '_Z/..TgFg0/ptQtpAgws'
        secret = 'test'
        self.assertTrue(handler.verify(secret, even_hash))
        self.assertTrue(handler.verify(secret, odd_hash))
        self.assertTrue(handler.needs_update(even_hash))
        self.assertFalse(handler.needs_update(odd_hash))
        new_hash = handler.hash('stub')
        self.assertFalse(handler.needs_update(new_hash))