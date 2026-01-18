import sys
from io import BytesIO
from .. import config, errors, gpg, tests, trace, ui
from . import TestCase, features
class TestDisabled(TestCase):

    def test_sign(self):
        self.assertRaises(gpg.SigningFailed, gpg.DisabledGPGStrategy(None).sign, b'content', gpg.MODE_CLEAR)

    def test_verify(self):
        self.assertRaises(gpg.SignatureVerificationFailed, gpg.DisabledGPGStrategy(None).verify, b'content')