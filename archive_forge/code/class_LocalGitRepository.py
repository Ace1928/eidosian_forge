from io import BytesIO
from dulwich.errors import NotCommitError
from dulwich.object_store import peel_sha, tree_lookup_path
from dulwich.objects import ZERO_SHA, Commit
from .. import check, errors
from .. import graph as _mod_graph
from .. import lock, repository
from .. import revision as _mod_revision
from .. import trace, transactions, ui
from ..decorators import only_raises
from ..foreign import ForeignRepository
from .filegraph import GitFileLastChangeScanner, GitFileParentProvider
from .mapping import (default_mapping, encode_git_path, foreign_vcs_git,
from .tree import GitRevisionTree
class LocalGitRepository(GitRepository):
    """Git repository on the file system."""

    def __init__(self, gitdir):
        GitRepository.__init__(self, gitdir)
        self._git = gitdir._git
        self._file_change_scanner = GitFileLastChangeScanner(self)
        self._transaction = None

    def get_commit_builder(self, branch, parents, config, timestamp=None, timezone=None, committer=None, revprops=None, revision_id=None, lossy=False):
        """Obtain a CommitBuilder for this repository.

        :param branch: Branch to commit to.
        :param parents: Revision ids of the parents of the new revision.
        :param config: Configuration to use.
        :param timestamp: Optional timestamp recorded for commit.
        :param timezone: Optional timezone for timestamp.
        :param committer: Optional committer to set for commit.
        :param revprops: Optional dictionary of revision properties.
        :param revision_id: Optional revision id.
        :param lossy: Whether to discard data that can not be natively
            represented, when pushing to a foreign VCS
        """
        from .commit import GitCommitBuilder
        builder = GitCommitBuilder(self, parents, config, timestamp, timezone, committer, revprops, revision_id, lossy)
        self.start_write_group()
        return builder

    def _write_git_config(self, cs):
        f = BytesIO()
        cs.write_to_file(f)
        self._git._put_named_file('config', f.getvalue())

    def get_file_graph(self):
        return _mod_graph.Graph(GitFileParentProvider(self._file_change_scanner))

    def iter_files_bytes(self, desired_files):
        """Iterate through file versions.

        Files will not necessarily be returned in the order they occur in
        desired_files.  No specific order is guaranteed.

        Yields pairs of identifier, bytes_iterator.  identifier is an opaque
        value supplied by the caller as part of desired_files.  It should
        uniquely identify the file version in the caller's context.  (Examples:
        an index number or a TreeTransform trans_id.)

        bytes_iterator is an iterable of bytestrings for the file.  The
        kind of iterable and length of the bytestrings are unspecified, but for
        this implementation, it is a list of bytes produced by
        VersionedFile.get_record_stream().

        :param desired_files: a list of (file_id, revision_id, identifier)
            triples
        """
        per_revision = {}
        for file_id, revision_id, identifier in desired_files:
            per_revision.setdefault(revision_id, []).append((file_id, identifier))
        for revid, files in per_revision.items():
            try:
                commit_id, mapping = self.lookup_bzr_revision_id(revid)
            except errors.NoSuchRevision:
                raise errors.RevisionNotPresent(revid, self)
            try:
                commit = self._git.object_store[commit_id]
            except KeyError:
                raise errors.RevisionNotPresent(revid, self)
            root_tree = commit.tree
            for fileid, identifier in files:
                try:
                    path = mapping.parse_file_id(fileid)
                except ValueError:
                    raise errors.RevisionNotPresent((fileid, revid), self)
                try:
                    mode, item_id = tree_lookup_path(self._git.object_store.__getitem__, root_tree, encode_git_path(path))
                    obj = self._git.object_store[item_id]
                except KeyError:
                    raise errors.RevisionNotPresent((fileid, revid), self)
                else:
                    if obj.type_name == b'tree':
                        yield (identifier, [])
                    elif obj.type_name == b'blob':
                        yield (identifier, obj.chunked)
                    else:
                        raise AssertionError('file text resolved to %r' % obj)

    def gather_stats(self, revid=None, committers=None):
        """See Repository.gather_stats()."""
        result = super().gather_stats(revid, committers)
        revs = []
        for sha in self._git.object_store:
            o = self._git.object_store[sha]
            if o.type_name == b'commit':
                revs.append(o.id)
        result['revisions'] = len(revs)
        return result

    def _iter_revision_ids(self):
        mapping = self.get_mapping()
        for sha in self._git.object_store:
            o = self._git.object_store[sha]
            if not isinstance(o, Commit):
                continue
            revid = mapping.revision_id_foreign_to_bzr(o.id)
            yield (o.id, revid)

    def all_revision_ids(self):
        ret = set()
        for git_sha, revid in self._iter_revision_ids():
            ret.add(revid)
        return list(ret)

    def _get_parents(self, revid, no_alternates=False):
        if type(revid) != bytes:
            raise ValueError
        try:
            hexsha, mapping = self.lookup_bzr_revision_id(revid)
        except errors.NoSuchRevision:
            return None
        try:
            commit = self._git.object_store[hexsha]
        except KeyError:
            return None
        ret = []
        for p in commit.parents:
            try:
                ret.append(self.lookup_foreign_revision_id(p, mapping))
            except KeyError:
                ret.append(mapping.revision_id_foreign_to_bzr(p))
        return ret

    def _get_parent_map_no_fallbacks(self, revids):
        return self.get_parent_map(revids, no_alternates=True)

    def get_parent_map(self, revids, no_alternates=False):
        parent_map = {}
        for revision_id in revids:
            parents = self._get_parents(revision_id, no_alternates=no_alternates)
            if revision_id == _mod_revision.NULL_REVISION:
                parent_map[revision_id] = ()
                continue
            if parents is None:
                continue
            if len(parents) == 0:
                parents = [_mod_revision.NULL_REVISION]
            parent_map[revision_id] = tuple(parents)
        return parent_map

    def get_known_graph_ancestry(self, revision_ids):
        """Return the known graph for a set of revision ids and their ancestors.
        """
        pending = set(revision_ids)
        parent_map = {}
        while pending:
            this_parent_map = {}
            for revid in pending:
                if revid == _mod_revision.NULL_REVISION:
                    continue
                parents = self._get_parents(revid)
                if parents is not None:
                    this_parent_map[revid] = parents
            parent_map.update(this_parent_map)
            pending = set()
            for values in this_parent_map.values():
                pending.update(values)
            pending = pending.difference(parent_map)
        return _mod_graph.KnownGraph(parent_map)

    def get_signature_text(self, revision_id):
        git_commit_id, mapping = self.lookup_bzr_revision_id(revision_id)
        try:
            commit = self._git.object_store[git_commit_id]
        except KeyError:
            raise errors.NoSuchRevision(self, revision_id)
        if commit.gpgsig is None:
            raise errors.NoSuchRevision(self, revision_id)
        return commit.gpgsig

    def check(self, revision_ids=None, callback_refs=None, check_repo=True):
        result = GitCheck(self, check_repo=check_repo)
        result.check(callback_refs)
        return result

    def pack(self, hint=None, clean_obsolete_packs=False):
        self._git.object_store.pack_loose_objects()

    def lookup_foreign_revision_id(self, foreign_revid, mapping=None):
        """Lookup a revision id.

        :param foreign_revid: Foreign revision id to look up
        :param mapping: Mapping to use (use default mapping if not specified)
        :raise KeyError: If foreign revision was not found
        :return: bzr revision id
        """
        if not isinstance(foreign_revid, bytes):
            raise TypeError(foreign_revid)
        if mapping is None:
            mapping = self.get_mapping()
        if foreign_revid == ZERO_SHA:
            return _mod_revision.NULL_REVISION
        unpeeled, peeled = peel_sha(self._git.object_store, foreign_revid)
        if not isinstance(peeled, Commit):
            raise NotCommitError(peeled.id)
        revid = mapping.get_revision_id(peeled)
        return revid

    def has_signature_for_revision_id(self, revision_id):
        """Check whether a GPG signature is present for this revision.

        This is never the case for Git repositories.
        """
        try:
            self.get_signature_text(revision_id)
        except errors.NoSuchRevision:
            return False
        else:
            return True

    def verify_revision_signature(self, revision_id, gpg_strategy):
        """Verify the signature on a revision.

        :param revision_id: the revision to verify
        :gpg_strategy: the GPGStrategy object to used

        :return: gpg.SIGNATURE_VALID or a failed SIGNATURE_ value
        """
        from breezy import gpg
        with self.lock_read():
            git_commit_id, mapping = self.lookup_bzr_revision_id(revision_id)
            try:
                commit = self._git.object_store[git_commit_id]
            except KeyError:
                raise errors.NoSuchRevision(self, revision_id)
            if commit.gpgsig is None:
                return (gpg.SIGNATURE_NOT_SIGNED, None)
            without_sig = Commit.from_string(commit.as_raw_string())
            without_sig.gpgsig = None
            result, key, plain_text = gpg_strategy.verify(without_sig.as_raw_string(), commit.gpgsig)
            return (result, key)

    def lookup_bzr_revision_id(self, bzr_revid, mapping=None):
        """Lookup a bzr revision id in a Git repository.

        :param bzr_revid: Bazaar revision id
        :param mapping: Optional mapping to use
        :return: Tuple with git commit id, mapping that was used and supplement
            details
        """
        try:
            git_sha, mapping = mapping_registry.revision_id_bzr_to_foreign(bzr_revid)
        except errors.InvalidRevisionId:
            raise errors.NoSuchRevision(self, bzr_revid)
        else:
            return (git_sha, mapping)

    def get_revision(self, revision_id):
        if not isinstance(revision_id, bytes):
            raise errors.InvalidRevisionId(revision_id, self)
        git_commit_id, mapping = self.lookup_bzr_revision_id(revision_id)
        try:
            commit = self._git.object_store[git_commit_id]
        except KeyError:
            raise errors.NoSuchRevision(self, revision_id)
        revision, roundtrip_revid, verifiers = mapping.import_commit(commit, self.lookup_foreign_revision_id, strict=False)
        if revision is None:
            raise AssertionError
        if roundtrip_revid:
            revision.revision_id = roundtrip_revid
        return revision

    def has_revision(self, revision_id):
        """See Repository.has_revision."""
        if revision_id == _mod_revision.NULL_REVISION:
            return True
        try:
            git_commit_id, mapping = self.lookup_bzr_revision_id(revision_id)
        except errors.NoSuchRevision:
            return False
        return git_commit_id in self._git

    def has_revisions(self, revision_ids):
        """See Repository.has_revisions."""
        return set(filter(self.has_revision, revision_ids))

    def iter_revisions(self, revision_ids):
        """See Repository.get_revisions."""
        for revid in revision_ids:
            try:
                rev = self.get_revision(revid)
            except errors.NoSuchRevision:
                rev = None
            yield (revid, rev)

    def revision_trees(self, revids):
        """See Repository.revision_trees."""
        for revid in revids:
            yield self.revision_tree(revid)

    def revision_tree(self, revision_id):
        """See Repository.revision_tree."""
        if revision_id is None:
            raise ValueError('invalid revision id %s' % revision_id)
        return GitRevisionTree(self, revision_id)

    def set_make_working_trees(self, trees):
        raise errors.UnsupportedOperation(self.set_make_working_trees, self)

    def make_working_trees(self):
        return not self._git.get_config().get_boolean(('core',), 'bare')