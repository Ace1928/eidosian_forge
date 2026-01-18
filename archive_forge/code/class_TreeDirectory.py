from typing import (TYPE_CHECKING, Dict, Iterator, List, Optional, Type, Union,
from . import errors, lock, osutils
from . import revision as _mod_revision
from . import trace
from .inter import InterObject
class TreeDirectory(TreeEntry):
    """See TreeEntry. This is a directory in a working tree."""
    __slots__: List[str] = []
    kind = 'directory'

    def kind_character(self):
        return '/'