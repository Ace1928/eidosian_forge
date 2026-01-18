from collections import deque
from ..lazy_import import lazy_import
from breezy.bzr import (
from .. import errors, lazy_regex, osutils, trace
from .static_tuple import StaticTuple
class InvalidEntryName(errors.InternalBzrError):
    _fmt = 'Invalid entry name: %(name)s'

    def __init__(self, name):
        errors.BzrError.__init__(self)
        self.name = name