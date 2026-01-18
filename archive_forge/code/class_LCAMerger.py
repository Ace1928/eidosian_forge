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
class LCAMerger(WeaveMerger):
    requires_file_merge_plan = True

    def _generate_merge_plan(self, this_path, base):
        return self.this_tree.plan_file_lca_merge(this_path, self.other_tree, base=base)