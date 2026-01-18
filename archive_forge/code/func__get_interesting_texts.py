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
def _get_interesting_texts(self, parent_map):
    """Return a dict of texts we are interested in.

        Note that the input is in key tuples, but the output is in plain
        revision ids.

        :param parent_map: The output from _find_recursive_lcas
        :return: A dict of {'revision_id':lines} as returned by
            _PlanMergeBase.get_lines()
        """
    all_revision_keys = set(parent_map)
    all_revision_keys.add(self.a_key)
    all_revision_keys.add(self.b_key)
    all_texts = self.get_lines([k[-1] for k in all_revision_keys])
    return all_texts