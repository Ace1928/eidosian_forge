from typing import (TYPE_CHECKING, Dict, Iterator, List, Optional, Type, Union,
from . import errors, lock, osutils
from . import revision as _mod_revision
from . import trace
from .inter import InterObject
class FileTimestampUnavailable(errors.BzrError):
    _fmt = 'The filestamp for %(path)s is not available.'
    internal_error = True

    def __init__(self, path):
        self.path = path