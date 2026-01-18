import contextlib
import tempfile
from typing import Type
from .lazy_import import lazy_import
import patiencediff
from breezy import (
from breezy.bzr import (
from breezy.i18n import gettext
from . import decorators, errors, hooks, osutils, registry
from . import revision as _mod_revision
from . import trace, transform
from . import transport as _mod_transport
from . import tree as _mod_tree
class _MergeTypeParameterizer:
    """Wrap a merge-type class to provide extra parameters.

    This is hack used by MergeIntoMerger to pass some extra parameters to its
    merge_type.  Merger.do_merge() sets up its own set of parameters to pass to
    the 'merge_type' member.  It is difficult override do_merge without
    re-writing the whole thing, so instead we create a wrapper which will pass
    the extra parameters.
    """

    def __init__(self, merge_type, **kwargs):
        self._extra_kwargs = kwargs
        self._merge_type = merge_type

    def __call__(self, *args, **kwargs):
        kwargs.update(self._extra_kwargs)
        return self._merge_type(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._merge_type, name)