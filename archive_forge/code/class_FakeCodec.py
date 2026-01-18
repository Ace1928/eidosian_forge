import codecs
import locale
import sys
from typing import Set
from .. import osutils
from . import TestCase
from .ui_testing import BytesIOWithEncoding, StringIOWithEncoding
class FakeCodec:
    """Special class that helps testing over several non-existed encodings.

    Clients can add new encoding names, but because of how codecs is
    implemented they cannot be removed. Be careful with naming to avoid
    collisions between tests.
    """
    _registered: bool = False
    _enabled_encodings: Set[str] = set()

    def add(self, encoding_name):
        """Adding encoding name to fake.

        :type   encoding_name:  lowercase plain string
        """
        if not self._registered:
            codecs.register(self)
            self._registered = True
        if encoding_name is not None:
            self._enabled_encodings.add(encoding_name)

    def __call__(self, encoding_name):
        """Called indirectly by codecs module during lookup"""
        if encoding_name in self._enabled_encodings:
            return codecs.lookup('latin-1')