from __future__ import unicode_literals
import typing
from collections import defaultdict, deque, namedtuple
from ._repr import make_repr
from .errors import FSError
from .path import abspath, combine, normpath
@classmethod
def _raise_errors(cls, path, error):
    """Re-raise dir scan errors when called."""
    return False