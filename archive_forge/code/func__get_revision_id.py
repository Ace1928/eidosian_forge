from contextlib import ExitStack
import time
from typing import Type
from breezy import registry
from breezy import revision as _mod_revision
from breezy.osutils import format_date, local_time_offset
def _get_revision_id(self):
    """Get the revision id we are working on."""
    if self._revision_id is not None:
        return self._revision_id
    if self._working_tree is not None:
        return self._working_tree.last_revision()
    return self._branch.last_revision()