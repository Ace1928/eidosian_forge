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
def _get_filter_tree_path(self, path):
    if self.this_tree.supports_content_filtering():
        filter_path = _mod_tree.find_previous_path(self.other_tree, self.working_tree, path)
        if filter_path is None:
            filter_path = path
        return filter_path
    return None