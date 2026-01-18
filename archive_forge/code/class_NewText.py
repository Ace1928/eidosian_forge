import errno
import os
from io import BytesIO
from .lazy_import import lazy_import
import gzip
import itertools
import patiencediff
from breezy import (
from . import errors
from .i18n import gettext
class NewText:
    """The contents of text that is introduced by this text"""
    __slots__ = ['lines']

    def __init__(self, lines):
        self.lines = lines

    def __eq__(self, other):
        if self.__class__ is not other.__class__:
            return False
        return other.lines == self.lines

    def __repr__(self):
        return 'NewText(%r)' % self.lines

    def to_patch(self):
        yield (b'i %d\n' % len(self.lines))
        yield from self.lines
        yield b'\n'