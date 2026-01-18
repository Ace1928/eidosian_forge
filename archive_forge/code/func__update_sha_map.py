import posixpath
import stat
from typing import Dict, Iterable, Iterator, List
from dulwich.object_store import BaseObjectStore
from dulwich.objects import (ZERO_SHA, Blob, Commit, ObjectID, ShaFile, Tree,
from dulwich.pack import Pack, PackData, pack_objects_to_data
from .. import errors, lru_cache, osutils, trace, ui
from ..bzr.testament import StrictTestament3
from ..lock import LogicalLockResult
from ..revision import NULL_REVISION
from ..tree import InterTree
from .cache import from_repository as cache_from_repository
from .mapping import (default_mapping, encode_git_path, entry_mode,
from .unpeel_map import UnpeelMap
def _update_sha_map(self, stop_revision=None):
    if not self.is_locked():
        raise errors.LockNotHeld(self)
    if self._map_updated:
        return
    if stop_revision is not None and (not self._missing_revisions([stop_revision])):
        return
    graph = self.repository.get_graph()
    if stop_revision is None:
        all_revids = self.repository.all_revision_ids()
        missing_revids = self._missing_revisions(all_revids)
    else:
        heads = {stop_revision}
        missing_revids = self._missing_revisions(heads)
        while heads:
            parents = graph.get_parent_map(heads)
            todo = set()
            for p in parents.values():
                todo.update([x for x in p if x not in missing_revids])
            heads = self._missing_revisions(todo)
            missing_revids.update(heads)
    if NULL_REVISION in missing_revids:
        missing_revids.remove(NULL_REVISION)
    missing_revids = self.repository.has_revisions(missing_revids)
    if not missing_revids:
        if stop_revision is None:
            self._map_updated = True
        return
    self.start_write_group()
    try:
        with ui.ui_factory.nested_progress_bar() as pb:
            for i, revid in enumerate(graph.iter_topo_order(missing_revids)):
                trace.mutter('processing %r', revid)
                pb.update('updating git map', i, len(missing_revids))
                self._update_sha_map_revision(revid)
        if stop_revision is None:
            self._map_updated = True
    except BaseException:
        self.abort_write_group()
        raise
    else:
        self.commit_write_group()