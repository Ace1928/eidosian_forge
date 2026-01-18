from typing import (TYPE_CHECKING, Dict, List, Optional, TextIO, Tuple, Union,
from .lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
import contextlib
import itertools
from . import config as _mod_config
from . import debug, errors, registry, repository
from . import revision as _mod_revision
from . import urlutils
from .controldir import (ControlComponent, ControlComponentFormat,
from .hooks import Hooks
from .inter import InterObject
from .lock import LogicalLockResult
from .revision import RevisionID
from .trace import is_quiet, mutter, mutter_callsite, note, warning
from .transport import Transport, get_transport
def _revision_relations(self, revision_a, revision_b, graph):
    """Determine the relationship between two revisions.

        Returns: One of: 'a_descends_from_b', 'b_descends_from_a', 'diverged'
        """
    heads = graph.heads([revision_a, revision_b])
    if heads == {revision_b}:
        return 'b_descends_from_a'
    elif heads == {revision_a, revision_b}:
        return 'diverged'
    elif heads == {revision_a}:
        return 'a_descends_from_b'
    else:
        raise AssertionError('invalid heads: {!r}'.format(heads))