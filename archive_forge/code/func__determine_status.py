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
def _determine_status(self, revision_id, unique_line_numbers):
    """Determines the status unique lines versus all lcas.

        Basically, determines why the line is unique to this revision.

        A line may be determined new, killed, or both.

        If a line is determined new, that means it was not present in at least
        one LCA, and is not present in the other merge revision.

        If a line is determined killed, that means the line was present in
        at least one LCA.

        If a line is killed and new, this indicates that the two merge
        revisions contain differing conflict resolutions.

        :param revision_id: The id of the revision in which the lines are
            unique
        :param unique_line_numbers: The line numbers of unique lines.
        :return: a tuple of (new_this, killed_other)
        """
    new = set()
    killed = set()
    unique_line_numbers = set(unique_line_numbers)
    for lca in self.lcas:
        blocks = self._get_matching_blocks(revision_id, lca)
        unique_vs_lca, _ignored = self._unique_lines(blocks)
        new.update(unique_line_numbers.intersection(unique_vs_lca))
        killed.update(unique_line_numbers.difference(unique_vs_lca))
    return (new, killed)