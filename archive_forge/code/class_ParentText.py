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
class ParentText:
    """A reference to text present in a parent text"""
    __slots__ = ['parent', 'parent_pos', 'child_pos', 'num_lines']

    def __init__(self, parent, parent_pos, child_pos, num_lines):
        self.parent = parent
        self.parent_pos = parent_pos
        self.child_pos = child_pos
        self.num_lines = num_lines

    def _as_dict(self):
        return {b'parent': self.parent, b'parent_pos': self.parent_pos, b'child_pos': self.child_pos, b'num_lines': self.num_lines}

    def __repr__(self):
        return 'ParentText(%(parent)r, %(parent_pos)r, %(child_pos)r, %(num_lines)r)' % self._as_dict()

    def __eq__(self, other):
        if self.__class__ is not other.__class__:
            return False
        return self._as_dict() == other._as_dict()

    def to_patch(self):
        yield (b'c %(parent)d %(parent_pos)d %(child_pos)d %(num_lines)d\n' % self._as_dict())