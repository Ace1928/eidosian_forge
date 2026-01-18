import enum
import re
import warnings
from git.cmd import handle_process_output
from git.compat import defenc
from git.objects.blob import Blob
from git.objects.util import mode_str_to_int
from git.util import finalize_process, hex_to_bin
from typing import (
from git.types import Literal, PathLike
@enum.unique
class DiffConstants(enum.Enum):
    """Special objects for :meth:`Diffable.diff`.

    See the :meth:`Diffable.diff` method's ``other`` parameter, which accepts various
    values including these.

    :note:
        These constants are also available as attributes of the :mod:`git.diff` module,
        the :class:`Diffable` class and its subclasses and instances, and the top-level
        :mod:`git` module.
    """
    NULL_TREE = enum.auto()
    'Stand-in indicating you want to compare against the empty tree in diffs.\n\n    Also accessible as :const:`git.NULL_TREE`, :const:`git.diff.NULL_TREE`, and\n    :const:`Diffable.NULL_TREE`.\n    '
    INDEX = enum.auto()
    'Stand-in indicating you want to diff against the index.\n\n    Also accessible as :const:`git.INDEX`, :const:`git.diff.INDEX`, and\n    :const:`Diffable.INDEX`, as well as :const:`Diffable.Index`. The latter has been\n    kept for backward compatibility and made an alias of this, so it may still be used.\n    '