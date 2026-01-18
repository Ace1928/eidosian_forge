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
class BazaarObjectStore(BaseObjectStore):
    """A Git-style object store backed onto a Bazaar repository."""

    def __init__(self, repository, mapping=None):
        self.repository = repository
        self._map_updated = False
        self._locked = None
        if mapping is None:
            self.mapping = default_mapping
        else:
            self.mapping = mapping
        self._cache = cache_from_repository(repository)
        self._content_cache_types = ('tree',)
        self.start_write_group = self._cache.idmap.start_write_group
        self.abort_write_group = self._cache.idmap.abort_write_group
        self.commit_write_group = self._cache.idmap.commit_write_group
        self.tree_cache = LRUTreeCache(self.repository)
        self.unpeel_map = UnpeelMap.from_repository(self.repository)

    def _missing_revisions(self, revisions):
        return self._cache.idmap.missing_revisions(revisions)

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

    def __iter__(self):
        self._update_sha_map()
        return iter(self._cache.idmap.sha1s())

    def _reconstruct_commit(self, rev, tree_sha, lossy, verifiers):
        """Reconstruct a Commit object.

        :param rev: Revision object
        :param tree_sha: SHA1 of the root tree object
        :param lossy: Whether or not to roundtrip bzr metadata
        :param verifiers: Verifiers for the commits
        :return: Commit object
        """

        def parent_lookup(revid):
            try:
                return self._lookup_revision_sha1(revid)
            except errors.NoSuchRevision:
                return None
        return self.mapping.export_commit(rev, tree_sha, parent_lookup, lossy, verifiers)

    def _revision_to_objects(self, rev, tree, lossy, add_cache_entry=None):
        """Convert a revision to a set of git objects.

        :param rev: Bazaar revision object
        :param tree: Bazaar revision tree
        :param lossy: Whether to not roundtrip all Bazaar revision data
        """
        unusual_modes = extract_unusual_modes(rev)
        present_parents = self.repository.has_revisions(rev.parent_ids)
        parent_trees = self.tree_cache.revision_trees([p for p in rev.parent_ids if p in present_parents])
        root_tree = None
        for path, obj, bzr_key_data in _tree_to_objects(tree, parent_trees, self._cache.idmap, unusual_modes, self.mapping.BZR_DUMMY_FILE, add_cache_entry):
            if path == '':
                root_tree = obj
                root_key_data = bzr_key_data
            else:
                yield (path, obj)
        if root_tree is None:
            if not rev.parent_ids:
                root_tree = Tree()
            else:
                base_sha1 = self._lookup_revision_sha1(rev.parent_ids[0])
                root_tree = self[self[base_sha1].tree]
            root_key_data = (tree.path2id(''), tree.get_revision_id())
        if add_cache_entry is not None:
            add_cache_entry(root_tree, root_key_data, '')
        yield ('', root_tree)
        if not lossy:
            testament3 = StrictTestament3(rev, tree)
            verifiers = {'testament3-sha1': testament3.as_sha1()}
        else:
            verifiers = {}
        commit_obj = self._reconstruct_commit(rev, root_tree.id, lossy=lossy, verifiers=verifiers)
        try:
            foreign_revid, mapping = mapping_registry.parse_revision_id(rev.revision_id)
        except errors.InvalidRevisionId:
            pass
        else:
            _check_expected_sha(foreign_revid, commit_obj)
        if add_cache_entry is not None:
            add_cache_entry(commit_obj, verifiers, None)
        yield (None, commit_obj)

    def _get_updater(self, rev):
        return self._cache.get_updater(rev)

    def _update_sha_map_revision(self, revid):
        rev = self.repository.get_revision(revid)
        tree = self.tree_cache.revision_tree(rev.revision_id)
        updater = self._get_updater(rev)
        for path, obj in self._revision_to_objects(rev, tree, lossy=not self.mapping.roundtripping, add_cache_entry=updater.add_object):
            if isinstance(obj, Commit):
                commit_obj = obj
        commit_obj = updater.finish()
        return commit_obj.id

    def iter_unpacked_subset(self, shas, *, include_comp=False, allow_missing: bool=False, convert_ofs_delta: bool=True) -> Iterator[ShaFile]:
        if not allow_missing and shas:
            raise KeyError(shas.pop())
        yield from []

    def _reconstruct_blobs(self, keys):
        """Return a Git Blob object from a fileid and revision stored in bzr.

        :param fileid: File id of the text
        :param revision: Revision of the text
        """
        stream = self.repository.iter_files_bytes(((key[0], key[1], key) for key in keys))
        for (file_id, revision, expected_sha), chunks in stream:
            blob = Blob()
            blob.chunked = list(chunks)
            if blob.id != expected_sha and blob.data == b'':
                tree = self.tree_cache.revision_tree(revision)
                path = tree.id2path(file_id)
                if tree.kind(path) == 'symlink':
                    blob = symlink_to_blob(tree.get_symlink_target(path))
            _check_expected_sha(expected_sha, blob)
            yield blob

    def _reconstruct_tree(self, fileid, revid, bzr_tree, unusual_modes, expected_sha=None):
        """Return a Git Tree object from a file id and a revision stored in bzr.

        :param fileid: fileid in the tree.
        :param revision: Revision of the tree.
        """

        def get_ie_sha1(path, entry):
            if entry.kind == 'directory':
                try:
                    return self._cache.idmap.lookup_tree_id(entry.file_id, revid)
                except (NotImplementedError, KeyError):
                    obj = self._reconstruct_tree(entry.file_id, revid, bzr_tree, unusual_modes)
                    if obj is None:
                        return None
                    else:
                        return obj.id
            elif entry.kind in ('file', 'symlink'):
                try:
                    return self._cache.idmap.lookup_blob_id(entry.file_id, entry.revision)
                except KeyError:
                    return next(self._reconstruct_blobs([(entry.file_id, entry.revision, None)])).id
            elif entry.kind == 'tree-reference':
                return self._lookup_revision_sha1(entry.reference_revision)
            else:
                raise AssertionError("unknown entry kind '%s'" % entry.kind)
        path = bzr_tree.id2path(fileid)
        tree = directory_to_tree(path, bzr_tree.iter_child_entries(path), get_ie_sha1, unusual_modes, self.mapping.BZR_DUMMY_FILE, bzr_tree.path2id('') == fileid)
        if tree is not None:
            _check_expected_sha(expected_sha, tree)
        return tree

    def get_parents(self, sha):
        """Retrieve the parents of a Git commit by SHA1.

        :param sha: SHA1 of the commit
        :raises: KeyError, NotCommitError
        """
        return self[sha].parents

    def _lookup_revision_sha1(self, revid):
        """Return the SHA1 matching a Bazaar revision."""
        if revid == NULL_REVISION:
            return ZERO_SHA
        try:
            return self._cache.idmap.lookup_commit(revid)
        except KeyError:
            try:
                return mapping_registry.parse_revision_id(revid)[0]
            except errors.InvalidRevisionId:
                self._update_sha_map(revid)
                return self._cache.idmap.lookup_commit(revid)

    def get_raw(self, sha):
        """Get the raw representation of a Git object by SHA1.

        :param sha: SHA1 of the git object
        """
        if len(sha) == 20:
            sha = sha_to_hex(sha)
        obj = self[sha]
        return (obj.type_num, obj.as_raw_string())

    def __contains__(self, sha):
        try:
            for type, type_data in self.lookup_git_sha(sha):
                if type == 'commit':
                    if self.repository.has_revision(type_data[0]):
                        return True
                elif type == 'blob':
                    if type_data in self.repository.texts:
                        return True
                elif type == 'tree':
                    if self.repository.has_revision(type_data[1]):
                        return True
                else:
                    raise AssertionError("Unknown object type '%s'" % type)
            else:
                return False
        except KeyError:
            return False

    def lock_read(self):
        self._locked = 'r'
        self._map_updated = False
        self.repository.lock_read()
        return LogicalLockResult(self.unlock)

    def lock_write(self):
        self._locked = 'r'
        self._map_updated = False
        self.repository.lock_write()
        return LogicalLockResult(self.unlock)

    def is_locked(self):
        return self._locked is not None

    def unlock(self):
        self._locked = None
        self._map_updated = False
        self.repository.unlock()

    def lookup_git_shas(self, shas: Iterable[ObjectID]) -> Dict[ObjectID, List]:
        ret: Dict[ObjectID, List] = {}
        for sha in shas:
            if sha == ZERO_SHA:
                ret[sha] = [('commit', (NULL_REVISION, None, {}))]
                continue
            try:
                ret[sha] = list(self._cache.idmap.lookup_git_sha(sha))
            except KeyError:
                self._update_sha_map()
                try:
                    ret[sha] = list(self._cache.idmap.lookup_git_sha(sha))
                except KeyError:
                    pass
        return ret

    def lookup_git_sha(self, sha):
        return self.lookup_git_shas([sha])[sha]

    def __getitem__(self, sha):
        with self.repository.lock_read():
            for kind, type_data in self.lookup_git_sha(sha):
                if kind == 'commit':
                    revid, tree_sha, verifiers = type_data
                    try:
                        rev = self.repository.get_revision(revid)
                    except errors.NoSuchRevision:
                        if revid == NULL_REVISION:
                            raise AssertionError('should not try to look up NULL_REVISION')
                        trace.mutter('entry for %s %s in shamap: %r, but not found in repository', kind, sha, type_data)
                        raise KeyError(sha)
                    commit = self._reconstruct_commit(rev, tree_sha, lossy=not self.mapping.roundtripping, verifiers=verifiers)
                    _check_expected_sha(sha, commit)
                    return commit
                elif kind == 'blob':
                    fileid, revision = type_data
                    blobs = self._reconstruct_blobs([(fileid, revision, sha)])
                    return next(blobs)
                elif kind == 'tree':
                    fileid, revid = type_data
                    try:
                        tree = self.tree_cache.revision_tree(revid)
                        rev = self.repository.get_revision(revid)
                    except errors.NoSuchRevision:
                        trace.mutter('entry for %s %s in shamap: %r, but not found in repository', kind, sha, type_data)
                        raise KeyError(sha)
                    unusual_modes = extract_unusual_modes(rev)
                    try:
                        return self._reconstruct_tree(fileid, revid, tree, unusual_modes, expected_sha=sha)
                    except errors.NoSuchRevision:
                        raise KeyError(sha)
                else:
                    raise AssertionError("Unknown object type '%s'" % kind)
            else:
                raise KeyError(sha)

    def generate_lossy_pack_data(self, have, want, shallow=None, progress=None, get_tagged=None, ofs_delta=False):
        object_ids = list(self.find_missing_objects(have, want, progress=progress, shallow=shallow, get_tagged=get_tagged, lossy=True))
        return pack_objects_to_data([(self[oid], path) for oid, (type_num, path) in object_ids])

    def find_missing_objects(self, have, want, shallow=None, progress=None, ofs_delta=False, get_tagged=None, lossy=False):
        """Iterate over the contents of a pack file.

        :param have: List of SHA1s of objects that should not be sent
        :param want: List of SHA1s of objects that should be sent
        """
        processed = set()
        ret: Dict[ObjectID, List] = self.lookup_git_shas(have + want)
        for commit_sha in have:
            commit_sha = self.unpeel_map.peel_tag(commit_sha, commit_sha)
            try:
                for type, type_data in ret[commit_sha]:
                    if type != 'commit':
                        raise AssertionError('Type was %s, not commit' % type)
                    processed.add(type_data[0])
            except KeyError:
                trace.mutter('unable to find remote ref %s', commit_sha)
        pending = set()
        for commit_sha in want:
            if commit_sha in have:
                continue
            try:
                for type, type_data in ret[commit_sha]:
                    if type != 'commit':
                        raise AssertionError('Type was %s, not commit' % type)
                    pending.add(type_data[0])
            except KeyError:
                pass
        shallows = set()
        for commit_sha in shallow or set():
            try:
                for type, type_data in ret[commit_sha]:
                    if type != 'commit':
                        raise AssertionError('Type was %s, not commit' % type)
                    shallows.add(type_data[0])
            except KeyError:
                pass
        seen = set()
        with self.repository.lock_read():
            graph = self.repository.get_graph()
            todo = _find_missing_bzr_revids(graph, pending, processed, shallow)
            with ui.ui_factory.nested_progress_bar() as pb:
                for i, revid in enumerate(graph.iter_topo_order(todo)):
                    pb.update('generating git objects', i, len(todo))
                    try:
                        rev = self.repository.get_revision(revid)
                    except errors.NoSuchRevision:
                        continue
                    tree = self.tree_cache.revision_tree(revid)
                    for path, obj in self._revision_to_objects(rev, tree, lossy=lossy):
                        if obj.id not in seen:
                            yield (obj.id, (obj.type_num, path))
                            seen.add(obj.id)

    def add_thin_pack(self):
        import os
        import tempfile
        fd, path = tempfile.mkstemp(suffix='.pack')
        f = os.fdopen(fd, 'wb')

        def commit():
            from .fetch import import_git_objects
            os.fsync(fd)
            f.close()
            if os.path.getsize(path) == 0:
                return
            pd = PackData(path)
            pd.create_index_v2(path[:-5] + '.idx', self.object_store.get_raw)
            p = Pack(path[:-5])
            with self.repository.lock_write():
                self.repository.start_write_group()
                try:
                    import_git_objects(self.repository, self.mapping, p.iterobjects(get_raw=self.get_raw), self.object_store)
                except BaseException:
                    self.repository.abort_write_group()
                    raise
                else:
                    self.repository.commit_write_group()
        return (f, commit)
    add_pack = add_thin_pack