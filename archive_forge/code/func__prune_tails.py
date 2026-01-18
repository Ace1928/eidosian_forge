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
@staticmethod
def _prune_tails(parent_map, child_map, tails_to_remove):
    """Remove tails from the parent map.

        This will remove the supplied revisions until no more children have 0
        parents.

        :param parent_map: A dict of {child: [parents]}, this dictionary will
            be modified in place.
        :param tails_to_remove: A list of tips that should be removed,
            this list will be consumed
        :param child_map: The reverse dict of parent_map ({parent: [children]})
            this dict will be modified
        :return: None, parent_map will be modified in place.
        """
    while tails_to_remove:
        next = tails_to_remove.pop()
        parent_map.pop(next)
        children = child_map.pop(next)
        for child in children:
            child_parents = parent_map[child]
            child_parents.remove(next)
            if len(child_parents) == 0:
                tails_to_remove.append(child)