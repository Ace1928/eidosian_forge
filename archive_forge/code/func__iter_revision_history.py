from contextlib import ExitStack
import time
from typing import Type
from breezy import registry
from breezy import revision as _mod_revision
from breezy.osutils import format_date, local_time_offset
def _iter_revision_history(self):
    """Find the messages for all revisions in history."""
    last_rev = self._get_revision_id()
    repository = self._branch.repository
    with repository.lock_read():
        graph = repository.get_graph()
        revhistory = list(graph.iter_lefthand_ancestry(last_rev, [_mod_revision.NULL_REVISION]))
        for revision_id in reversed(revhistory):
            rev = repository.get_revision(revision_id)
            yield (rev.revision_id, rev.message, rev.timestamp, rev.timezone)