import bz2
import os
import re
import sys
import zlib
from typing import Callable, List, Optional
import fastbencode as bencode
from .. import branch
from .. import bzr as _mod_bzr
from .. import config as _mod_config
from .. import (controldir, debug, errors, gpg, graph, lock, lockdir, osutils,
from .. import repository as _mod_repository
from .. import revision as _mod_revision
from .. import transport as _mod_transport
from .. import ui, urlutils
from ..branch import BranchWriteLockResult
from ..decorators import only_raises
from ..errors import NoSuchRevision, SmartProtocolError
from ..i18n import gettext
from ..lockable_files import LockableFiles
from ..repository import RepositoryWriteLockResult, _LazyListJoin
from ..revision import NULL_REVISION
from ..trace import log_exception_quietly, mutter, note, warning
from . import branch as bzrbranch
from . import bzrdir as _mod_bzrdir
from . import inventory_delta
from . import repository as bzrrepository
from . import testament as _mod_testament
from . import vf_repository, vf_search
from .branch import BranchReferenceFormat
from .inventory import Inventory
from .inventorytree import InventoryRevisionTree
from .serializer import format_registry as serializer_format_registry
from .smart import client
from .smart import repository as smart_repo
from .smart import vfs
from .smart.client import _SmartClient
from .versionedfile import FulltextContentFactory
class RemoteRepository(_mod_repository.Repository, _RpcHelper, lock._RelockDebugMixin):
    """Repository accessed over rpc.

    For the moment most operations are performed using local transport-backed
    Repository objects.
    """
    _format: RemoteRepositoryFormat
    _real_repository: Optional[_mod_repository.Repository]

    def __init__(self, remote_bzrdir: RemoteBzrDir, format: RemoteRepositoryFormat, real_repository: Optional[_mod_repository.Repository]=None, _client=None):
        """Create a RemoteRepository instance.

        :param remote_bzrdir: The bzrdir hosting this repository.
        :param format: The RemoteFormat object to use.
        :param real_repository: If not None, a local implementation of the
            repository logic for the repository, usually accessing the data
            via the VFS.
        :param _client: Private testing parameter - override the smart client
            to be used by the repository.
        """
        if real_repository:
            self._real_repository = real_repository
        else:
            self._real_repository = None
        self.controldir = remote_bzrdir
        if _client is None:
            self._client = remote_bzrdir._client
        else:
            self._client = _client
        self._format = format
        self._lock_mode = None
        self._lock_token = None
        self._write_group_tokens = None
        self._lock_count = 0
        self._leave_lock = False
        self._unstacked_provider = graph.CachingParentsProvider(get_parent_map=self._get_parent_map_rpc)
        self._unstacked_provider.disable_cache()
        self._reconcile_does_inventory_gc = False
        self._reconcile_fixes_text_parents = False
        self._reconcile_backsup_inventory = False
        self.base = self.controldir.transport.base
        self._fallback_repositories = []

    @property
    def user_transport(self):
        return self.controldir.user_transport

    @property
    def control_transport(self):
        return self.controldir.get_repository_transport(None)

    def __str__(self):
        return '{}({})'.format(self.__class__.__name__, self.base)
    __repr__ = __str__

    def abort_write_group(self, suppress_errors=False):
        """Complete a write group on the decorated repository.

        Smart methods perform operations in a single step so this API
        is not really applicable except as a compatibility thunk
        for older plugins that don't use e.g. the CommitBuilder
        facility.

        :param suppress_errors: see Repository.abort_write_group.
        """
        if self._real_repository:
            self._ensure_real()
            return self._real_repository.abort_write_group(suppress_errors=suppress_errors)
        if not self.is_in_write_group():
            if suppress_errors:
                mutter('(suppressed) not in write group')
                return
            raise errors.BzrError('not in write group')
        path = self.controldir._path_for_remote_call(self._client)
        try:
            response = self._call(b'Repository.abort_write_group', path, self._lock_token, [token.encode('utf-8') for token in self._write_group_tokens])
        except Exception as exc:
            self._write_group = None
            if not suppress_errors:
                raise
            mutter('abort_write_group failed')
            log_exception_quietly()
            note(gettext('bzr: ERROR (ignored): %s'), exc)
        else:
            if response != (b'ok',):
                raise errors.UnexpectedSmartServerResponse(response)
            self._write_group_tokens = None

    @property
    def chk_bytes(self):
        """Decorate the real repository for now.

        In the long term a full blown network facility is needed to avoid
        creating a real repository object locally.
        """
        self._ensure_real()
        return self._real_repository.chk_bytes

    def commit_write_group(self):
        """Complete a write group on the decorated repository.

        Smart methods perform operations in a single step so this API
        is not really applicable except as a compatibility thunk
        for older plugins that don't use e.g. the CommitBuilder
        facility.
        """
        if self._real_repository:
            self._ensure_real()
            return self._real_repository.commit_write_group()
        if not self.is_in_write_group():
            raise errors.BzrError('not in write group')
        path = self.controldir._path_for_remote_call(self._client)
        response = self._call(b'Repository.commit_write_group', path, self._lock_token, [token.encode('utf-8') for token in self._write_group_tokens])
        if response != (b'ok',):
            raise errors.UnexpectedSmartServerResponse(response)
        self._write_group_tokens = None
        self.refresh_data()

    def resume_write_group(self, tokens):
        if self._real_repository:
            return self._real_repository.resume_write_group(tokens)
        path = self.controldir._path_for_remote_call(self._client)
        try:
            response = self._call(b'Repository.check_write_group', path, self._lock_token, [token.encode('utf-8') for token in tokens])
        except errors.UnknownSmartMethod:
            self._ensure_real()
            return self._real_repository.resume_write_group(tokens)
        if response != (b'ok',):
            raise errors.UnexpectedSmartServerResponse(response)
        self._write_group_tokens = tokens

    def suspend_write_group(self):
        if self._real_repository:
            return self._real_repository.suspend_write_group()
        ret = self._write_group_tokens or []
        self._write_group_tokens = None
        return ret

    def get_missing_parent_inventories(self, check_for_missing_texts=True):
        self._ensure_real()
        return self._real_repository.get_missing_parent_inventories(check_for_missing_texts=check_for_missing_texts)

    def _get_rev_id_for_revno_vfs(self, revno, known_pair):
        self._ensure_real()
        return self._real_repository.get_rev_id_for_revno(revno, known_pair)

    def get_rev_id_for_revno(self, revno, known_pair):
        """See Repository.get_rev_id_for_revno."""
        path = self.controldir._path_for_remote_call(self._client)
        try:
            if self._client._medium._is_remote_before((1, 17)):
                return self._get_rev_id_for_revno_vfs(revno, known_pair)
            response = self._call(b'Repository.get_rev_id_for_revno', path, revno, known_pair)
        except errors.UnknownSmartMethod:
            self._client._medium._remember_remote_is_before((1, 17))
            return self._get_rev_id_for_revno_vfs(revno, known_pair)
        except UnknownErrorFromSmartServer as e:
            if len(e.error_tuple) < 3:
                raise
            if e.error_tuple[:2] != (b'error', b'ValueError'):
                raise
            m = re.match(b'requested revno \\(([0-9]+)\\) is later than given known revno \\(([0-9]+)\\)', e.error_tuple[2])
            if not m:
                raise
            raise errors.RevnoOutOfBounds(int(m.group(1)), (0, int(m.group(2))))
        if response[0] == b'ok':
            return (True, response[1])
        elif response[0] == b'history-incomplete':
            known_pair = response[1:3]
            for fallback in self._fallback_repositories:
                found, result = fallback.get_rev_id_for_revno(revno, known_pair)
                if found:
                    return (True, result)
                else:
                    known_pair = result
            return (False, known_pair)
        else:
            raise errors.UnexpectedSmartServerResponse(response)

    def _ensure_real(self):
        """Ensure that there is a _real_repository set.

        Used before calls to self._real_repository.

        Note that _ensure_real causes many roundtrips to the server which are
        not desirable, and prevents the use of smart one-roundtrip RPC's to
        perform complex operations (such as accessing parent data, streaming
        revisions etc). Adding calls to _ensure_real should only be done when
        bringing up new functionality, adding fallbacks for smart methods that
        require a fallback path, and never to replace an existing smart method
        invocation. If in doubt chat to the bzr network team.
        """
        if self._real_repository is None:
            if 'hpssvfs' in debug.debug_flags:
                import traceback
                warning('VFS Repository access triggered\n%s', ''.join(traceback.format_stack()))
            self._unstacked_provider.missing_keys.clear()
            self.controldir._ensure_real()
            self._set_real_repository(self.controldir._real_bzrdir.open_repository())

    def _translate_error(self, err, **context):
        self.controldir._translate_error(err, repository=self, **context)

    def find_text_key_references(self):
        """Find the text key references within the repository.

        :return: A dictionary mapping text keys ((fileid, revision_id) tuples)
            to whether they were referred to by the inventory of the
            revision_id that they contain. The inventory texts from all present
            revision ids are assessed to generate this report.
        """
        self._ensure_real()
        return self._real_repository.find_text_key_references()

    def _generate_text_key_index(self):
        """Generate a new text key index for the repository.

        This is an expensive function that will take considerable time to run.

        :return: A dict mapping (file_id, revision_id) tuples to a list of
            parents, also (file_id, revision_id) tuples.
        """
        self._ensure_real()
        return self._real_repository._generate_text_key_index()

    def _get_revision_graph(self, revision_id):
        """Private method for using with old (< 1.2) servers to fallback."""
        if revision_id is None:
            revision_id = b''
        elif _mod_revision.is_null(revision_id):
            return {}
        path = self.controldir._path_for_remote_call(self._client)
        response = self._call_expecting_body(b'Repository.get_revision_graph', path, revision_id)
        response_tuple, response_handler = response
        if response_tuple[0] != b'ok':
            raise errors.UnexpectedSmartServerResponse(response_tuple)
        coded = response_handler.read_body_bytes()
        if coded == b'':
            return {}
        lines = coded.split(b'\n')
        revision_graph = {}
        for line in lines:
            d = tuple(line.split())
            revision_graph[d[0]] = d[1:]
        return revision_graph

    def _get_sink(self):
        """See Repository._get_sink()."""
        return RemoteStreamSink(self)

    def _get_source(self, to_format):
        """Return a source for streaming from this repository."""
        return RemoteStreamSource(self, to_format)

    def get_file_graph(self):
        with self.lock_read():
            return graph.Graph(self.texts)

    def has_revision(self, revision_id):
        """True if this repository has a copy of the revision."""
        with self.lock_read():
            return revision_id in self.has_revisions((revision_id,))

    def has_revisions(self, revision_ids):
        """Probe to find out the presence of multiple revisions.

        :param revision_ids: An iterable of revision_ids.
        :return: A set of the revision_ids that were present.
        """
        with self.lock_read():
            parent_map = self.get_parent_map(revision_ids)
            result = set(parent_map)
            if _mod_revision.NULL_REVISION in revision_ids:
                result.add(_mod_revision.NULL_REVISION)
            return result

    def _has_same_fallbacks(self, other_repo):
        """Returns true if the repositories have the same fallbacks."""
        my_fb = self._fallback_repositories
        other_fb = other_repo._fallback_repositories
        if len(my_fb) != len(other_fb):
            return False
        for f, g in zip(my_fb, other_fb):
            if not f.has_same_location(g):
                return False
        return True

    def has_same_location(self, other):
        return self.__class__ is other.__class__ and self.controldir.transport.base == other.controldir.transport.base

    def get_graph(self, other_repository=None):
        """Return the graph for this repository format"""
        parents_provider = self._make_parents_provider(other_repository)
        return graph.Graph(parents_provider)

    def get_known_graph_ancestry(self, revision_ids):
        """Return the known graph for a set of revision ids and their ancestors.
        """
        with self.lock_read():
            revision_graph = {key: value for key, value in self.get_graph().iter_ancestry(revision_ids) if value is not None}
            revision_graph = _mod_repository._strip_NULL_ghosts(revision_graph)
            return graph.KnownGraph(revision_graph)

    def gather_stats(self, revid=None, committers=None):
        """See Repository.gather_stats()."""
        path = self.controldir._path_for_remote_call(self._client)
        if revid is None or _mod_revision.is_null(revid):
            fmt_revid = b''
        else:
            fmt_revid = revid
        if committers is None or not committers:
            fmt_committers = b'no'
        else:
            fmt_committers = b'yes'
        response_tuple, response_handler = self._call_expecting_body(b'Repository.gather_stats', path, fmt_revid, fmt_committers)
        if response_tuple[0] != b'ok':
            raise errors.UnexpectedSmartServerResponse(response_tuple)
        body = response_handler.read_body_bytes()
        result = {}
        for line in body.split(b'\n'):
            if not line:
                continue
            key, val_text = line.split(b':')
            key = key.decode('ascii')
            if key in ('revisions', 'size', 'committers'):
                result[key] = int(val_text)
            elif key in ('firstrev', 'latestrev'):
                values = val_text.split(b' ')[1:]
                result[key] = (float(values[0]), int(values[1]))
        return result

    def find_branches(self, using=False):
        """See Repository.find_branches()."""
        self._ensure_real()
        return self._real_repository.find_branches(using=using)

    def get_physical_lock_status(self):
        """See Repository.get_physical_lock_status()."""
        path = self.controldir._path_for_remote_call(self._client)
        try:
            response = self._call(b'Repository.get_physical_lock_status', path)
        except errors.UnknownSmartMethod:
            self._ensure_real()
            return self._real_repository.get_physical_lock_status()
        if response[0] not in (b'yes', b'no'):
            raise errors.UnexpectedSmartServerResponse(response)
        return response[0] == b'yes'

    def is_in_write_group(self):
        """Return True if there is an open write group.

        write groups are only applicable locally for the smart server..
        """
        if self._write_group_tokens is not None:
            return True
        if self._real_repository:
            return self._real_repository.is_in_write_group()

    def is_locked(self):
        return self._lock_count >= 1

    def is_shared(self):
        """See Repository.is_shared()."""
        path = self.controldir._path_for_remote_call(self._client)
        response = self._call(b'Repository.is_shared', path)
        if response[0] not in (b'yes', b'no'):
            raise SmartProtocolError('unexpected response code {}'.format(response))
        return response[0] == b'yes'

    def is_write_locked(self):
        return self._lock_mode == 'w'

    def _warn_if_deprecated(self, branch=None):
        pass

    def lock_read(self):
        """Lock the repository for read operations.

        :return: A breezy.lock.LogicalLockResult.
        """
        if not self._lock_mode:
            self._note_lock('r')
            self._lock_mode = 'r'
            self._lock_count = 1
            self._unstacked_provider.enable_cache(cache_misses=True)
            if self._real_repository is not None:
                self._real_repository.lock_read()
            for repo in self._fallback_repositories:
                repo.lock_read()
        else:
            self._lock_count += 1
        return lock.LogicalLockResult(self.unlock)

    def _remote_lock_write(self, token):
        path = self.controldir._path_for_remote_call(self._client)
        if token is None:
            token = b''
        err_context = {'token': token}
        response = self._call(b'Repository.lock_write', path, token, **err_context)
        if response[0] == b'ok':
            ok, token = response
            return token
        else:
            raise errors.UnexpectedSmartServerResponse(response)

    def lock_write(self, token=None, _skip_rpc=False):
        if not self._lock_mode:
            self._note_lock('w')
            if _skip_rpc:
                if self._lock_token is not None:
                    if token != self._lock_token:
                        raise errors.TokenMismatch(token, self._lock_token)
                self._lock_token = token
            else:
                self._lock_token = self._remote_lock_write(token)
            if self._real_repository is not None:
                self._real_repository.lock_write(token=self._lock_token)
            if token is not None:
                self._leave_lock = True
            else:
                self._leave_lock = False
            self._lock_mode = 'w'
            self._lock_count = 1
            cache_misses = self._real_repository is None
            self._unstacked_provider.enable_cache(cache_misses=cache_misses)
            for repo in self._fallback_repositories:
                repo.lock_read()
        elif self._lock_mode == 'r':
            raise errors.ReadOnlyError(self)
        else:
            self._lock_count += 1
        return RepositoryWriteLockResult(self.unlock, self._lock_token or None)

    def leave_lock_in_place(self):
        if not self._lock_token:
            raise NotImplementedError(self.leave_lock_in_place)
        self._leave_lock = True

    def dont_leave_lock_in_place(self):
        if not self._lock_token:
            raise NotImplementedError(self.dont_leave_lock_in_place)
        self._leave_lock = False

    def _set_real_repository(self, repository: _mod_repository.Repository):
        """Set the _real_repository for this repository.

        :param repository: The repository to fallback to for non-hpss
            implemented operations.
        """
        if self._real_repository is not None:
            if self.is_locked():
                raise AssertionError('_real_repository is already set')
        if isinstance(repository, RemoteRepository):
            raise AssertionError()
        self._real_repository = repository
        if self._fallback_repositories and len(self._real_repository._fallback_repositories) != len(self._fallback_repositories):
            if len(self._real_repository._fallback_repositories):
                raise AssertionError('cannot cleanly remove existing _fallback_repositories')
        for fb in self._fallback_repositories:
            self._real_repository.add_fallback_repository(fb)
        if self._lock_mode == 'w':
            self._real_repository.lock_write(self._lock_token)
        elif self._lock_mode == 'r':
            self._real_repository.lock_read()
        if self._write_group_tokens is not None:
            self._real_repository.resume_write_group(self._write_group_tokens)
            self._write_group_tokens = None

    def start_write_group(self):
        """Start a write group on the decorated repository.

        Smart methods perform operations in a single step so this API
        is not really applicable except as a compatibility thunk
        for older plugins that don't use e.g. the CommitBuilder
        facility.
        """
        if self._real_repository:
            self._ensure_real()
            return self._real_repository.start_write_group()
        if not self.is_write_locked():
            raise errors.NotWriteLocked(self)
        if self._write_group_tokens is not None:
            raise errors.BzrError('already in a write group')
        path = self.controldir._path_for_remote_call(self._client)
        try:
            response = self._call(b'Repository.start_write_group', path, self._lock_token)
        except (errors.UnknownSmartMethod, errors.UnsuspendableWriteGroup):
            self._ensure_real()
            return self._real_repository.start_write_group()
        if response[0] != b'ok':
            raise errors.UnexpectedSmartServerResponse(response)
        self._write_group_tokens = [token.decode('utf-8') for token in response[1]]

    def _unlock(self, token):
        path = self.controldir._path_for_remote_call(self._client)
        if not token:
            return
        err_context = {'token': token}
        response = self._call(b'Repository.unlock', path, token, **err_context)
        if response == (b'ok',):
            return
        else:
            raise errors.UnexpectedSmartServerResponse(response)

    @only_raises(errors.LockNotHeld, errors.LockBroken)
    def unlock(self):
        if not self._lock_count:
            return lock.cant_unlock_not_held(self)
        self._lock_count -= 1
        if self._lock_count > 0:
            return
        self._unstacked_provider.disable_cache()
        old_mode = self._lock_mode
        self._lock_mode = None
        try:
            if self._real_repository is not None:
                self._real_repository.unlock()
            elif self._write_group_tokens is not None:
                self.abort_write_group()
        finally:
            if old_mode == 'w':
                old_token = self._lock_token
                self._lock_token = None
                if not self._leave_lock:
                    self._unlock(old_token)
        for repo in self._fallback_repositories:
            repo.unlock()

    def break_lock(self):
        path = self.controldir._path_for_remote_call(self._client)
        try:
            response = self._call(b'Repository.break_lock', path)
        except errors.UnknownSmartMethod:
            self._ensure_real()
            return self._real_repository.break_lock()
        if response != (b'ok',):
            raise errors.UnexpectedSmartServerResponse(response)

    def _get_tarball(self, compression):
        """Return a TemporaryFile containing a repository tarball.

        Returns None if the server does not support sending tarballs.
        """
        import tempfile
        path = self.controldir._path_for_remote_call(self._client)
        try:
            response, protocol = self._call_expecting_body(b'Repository.tarball', path, compression.encode('ascii'))
        except errors.UnknownSmartMethod:
            protocol.cancel_read_body()
            return None
        if response[0] == b'ok':
            t = tempfile.NamedTemporaryFile()
            t.write(protocol.read_body_bytes())
            t.seek(0)
            return t
        raise errors.UnexpectedSmartServerResponse(response)

    def sprout(self, to_bzrdir, revision_id=None):
        """Create a descendent repository for new development.

        Unlike clone, this does not copy the settings of the repository.
        """
        with self.lock_read():
            dest_repo = self._create_sprouting_repo(to_bzrdir, shared=False)
            dest_repo.fetch(self, revision_id=revision_id)
            return dest_repo

    def _create_sprouting_repo(self, a_controldir, shared):
        if not isinstance(a_controldir._format, self.controldir._format.__class__):
            dest_repo = a_controldir.create_repository()
        else:
            try:
                dest_repo = self._format.initialize(a_controldir, shared=shared)
            except errors.UninitializableFormat:
                dest_repo = a_controldir.open_repository()
        return dest_repo

    def revision_tree(self, revision_id):
        with self.lock_read():
            if revision_id == _mod_revision.NULL_REVISION:
                return InventoryRevisionTree(self, Inventory(root_id=None), _mod_revision.NULL_REVISION)
            else:
                return list(self.revision_trees([revision_id]))[0]

    def get_serializer_format(self):
        path = self.controldir._path_for_remote_call(self._client)
        try:
            response = self._call(b'VersionedFileRepository.get_serializer_format', path)
        except errors.UnknownSmartMethod:
            self._ensure_real()
            return self._real_repository.get_serializer_format()
        if response[0] != b'ok':
            raise errors.UnexpectedSmartServerResponse(response)
        return response[1]

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
        if self._fallback_repositories and (not self._format.supports_chks):
            raise errors.BzrError('Cannot commit directly to a stacked branch in pre-2a formats. See https://bugs.launchpad.net/bzr/+bug/375013 for details.')
        commit_builder_kls = vf_repository.VersionedFileCommitBuilder
        result = commit_builder_kls(self, parents, config, timestamp, timezone, committer, revprops, revision_id, lossy)
        self.start_write_group()
        return result

    def add_fallback_repository(self, repository):
        """Add a repository to use for looking up data not held locally.

        :param repository: A repository.
        """
        if not self._format.supports_external_lookups:
            raise errors.UnstackableRepositoryFormat(self._format.network_name(), self.base)
        self._check_fallback_repository(repository)
        if self.is_locked():
            repository.lock_read()
        self._fallback_repositories.append(repository)
        if self._real_repository is not None:
            fallback_locations = [repo.user_url for repo in self._real_repository._fallback_repositories]
            if repository.user_url not in fallback_locations:
                self._real_repository.add_fallback_repository(repository)

    def _check_fallback_repository(self, repository):
        """Check that this repository can fallback to repository safely.

        Raise an error if not.

        :param repository: A repository to fallback to.
        """
        return _mod_repository.InterRepository._assert_same_model(self, repository)

    def add_inventory(self, revid, inv, parents):
        self._ensure_real()
        return self._real_repository.add_inventory(revid, inv, parents)

    def add_inventory_by_delta(self, basis_revision_id, delta, new_revision_id, parents, basis_inv=None, propagate_caches=False):
        self._ensure_real()
        return self._real_repository.add_inventory_by_delta(basis_revision_id, delta, new_revision_id, parents, basis_inv=basis_inv, propagate_caches=propagate_caches)

    def add_revision(self, revision_id, rev, inv=None):
        _mod_revision.check_not_reserved_id(revision_id)
        key = (revision_id,)
        if not self.inventories.get_parent_map([key]):
            if inv is None:
                raise errors.WeaveRevisionNotPresent(revision_id, self.inventories)
            else:
                rev.inventory_sha1 = self.add_inventory(revision_id, inv, rev.parent_ids)
        else:
            rev.inventory_sha1 = self.inventories.get_sha1s([key])[key]
        self._add_revision(rev)

    def _add_revision(self, rev):
        if self._real_repository is not None:
            return self._real_repository._add_revision(rev)
        lines = self._serializer.write_revision_to_lines(rev)
        key = (rev.revision_id,)
        parents = tuple(((parent,) for parent in rev.parent_ids))
        self._write_group_tokens, missing_keys = self._get_sink().insert_stream([('revisions', [ChunkedContentFactory(key, parents, None, lines, chunks_are_lines=True)])], self._format, self._write_group_tokens)

    def get_inventory(self, revision_id):
        with self.lock_read():
            return list(self.iter_inventories([revision_id]))[0]

    def _iter_inventories_rpc(self, revision_ids, ordering):
        if ordering is None:
            ordering = 'unordered'
        path = self.controldir._path_for_remote_call(self._client)
        body = b'\n'.join(revision_ids)
        response_tuple, response_handler = self._call_with_body_bytes_expecting_body(b'VersionedFileRepository.get_inventories', (path, ordering.encode('ascii')), body)
        if response_tuple[0] != b'ok':
            raise errors.UnexpectedSmartServerResponse(response_tuple)
        deserializer = inventory_delta.InventoryDeltaDeserializer()
        byte_stream = response_handler.read_streamed_body()
        decoded = smart_repo._byte_stream_to_stream(byte_stream)
        if decoded is None:
            return
        src_format, stream = decoded
        if src_format.network_name() != self._format.network_name():
            raise AssertionError('Mismatched RemoteRepository and stream src {!r}, {!r}'.format(src_format.network_name(), self._format.network_name()))
        prev_inv = Inventory(root_id=None, revision_id=_mod_revision.NULL_REVISION)
        try:
            substream_kind, substream = next(stream)
        except StopIteration:
            return
        if substream_kind != 'inventory-deltas':
            raise AssertionError('Unexpected stream %r received' % substream_kind)
        for record in substream:
            parent_id, new_id, versioned_root, tree_references, invdelta = deserializer.parse_text_bytes(record.get_bytes_as('lines'))
            if parent_id != prev_inv.revision_id:
                raise AssertionError('invalid base {!r} != {!r}'.format(parent_id, prev_inv.revision_id))
            inv = prev_inv.create_by_apply_delta(invdelta, new_id)
            yield (inv, inv.revision_id)
            prev_inv = inv

    def _iter_inventories_vfs(self, revision_ids, ordering=None):
        self._ensure_real()
        return self._real_repository._iter_inventories(revision_ids, ordering)

    def iter_inventories(self, revision_ids, ordering=None):
        """Get many inventories by revision_ids.

        This will buffer some or all of the texts used in constructing the
        inventories in memory, but will only parse a single inventory at a
        time.

        :param revision_ids: The expected revision ids of the inventories.
        :param ordering: optional ordering, e.g. 'topological'.  If not
            specified, the order of revision_ids will be preserved (by
            buffering if necessary).
        :return: An iterator of inventories.
        """
        if None in revision_ids or _mod_revision.NULL_REVISION in revision_ids:
            raise ValueError('cannot get null revision inventory')
        for inv, revid in self._iter_inventories(revision_ids, ordering):
            if inv is None:
                raise errors.NoSuchRevision(self, revid)
            yield inv

    def _iter_inventories(self, revision_ids, ordering=None):
        if len(revision_ids) == 0:
            return
        missing = set(revision_ids)
        if ordering is None:
            order_as_requested = True
            invs = {}
            order = list(revision_ids)
            order.reverse()
            next_revid = order.pop()
        else:
            order_as_requested = False
            if ordering != 'unordered' and self._fallback_repositories:
                raise ValueError('unsupported ordering %r' % ordering)
        iter_inv_fns = [self._iter_inventories_rpc] + [fallback._iter_inventories for fallback in self._fallback_repositories]
        try:
            for iter_inv in iter_inv_fns:
                request = [revid for revid in revision_ids if revid in missing]
                for inv, revid in iter_inv(request, ordering):
                    if inv is None:
                        continue
                    missing.remove(inv.revision_id)
                    if ordering != 'unordered':
                        invs[revid] = inv
                    else:
                        yield (inv, revid)
                if order_as_requested:
                    while next_revid in invs:
                        inv = invs.pop(next_revid)
                        yield (inv, inv.revision_id)
                        try:
                            next_revid = order.pop()
                        except IndexError:
                            next_revid = None
                            break
        except errors.UnknownSmartMethod:
            for inv, revid in self._iter_inventories_vfs(revision_ids, ordering):
                yield (inv, revid)
            return
        if order_as_requested:
            if next_revid is not None:
                yield (None, next_revid)
            while order:
                revid = order.pop()
                yield (invs.get(revid), revid)
        else:
            while missing:
                yield (None, missing.pop())

    def get_revision(self, revision_id):
        with self.lock_read():
            return self.get_revisions([revision_id])[0]

    def get_transaction(self):
        self._ensure_real()
        return self._real_repository.get_transaction()

    def clone(self, a_controldir, revision_id=None):
        with self.lock_read():
            dest_repo = self._create_sprouting_repo(a_controldir, shared=self.is_shared())
            self.copy_content_into(dest_repo, revision_id)
            return dest_repo

    def make_working_trees(self):
        """See Repository.make_working_trees"""
        path = self.controldir._path_for_remote_call(self._client)
        try:
            response = self._call(b'Repository.make_working_trees', path)
        except errors.UnknownSmartMethod:
            self._ensure_real()
            return self._real_repository.make_working_trees()
        if response[0] not in (b'yes', b'no'):
            raise SmartProtocolError('unexpected response code {}'.format(response))
        return response[0] == b'yes'

    def refresh_data(self):
        """Re-read any data needed to synchronise with disk.

        This method is intended to be called after another repository instance
        (such as one used by a smart server) has inserted data into the
        repository. On all repositories this will work outside of write groups.
        Some repository formats (pack and newer for breezy native formats)
        support refresh_data inside write groups. If called inside a write
        group on a repository that does not support refreshing in a write group
        IsInWriteGroupError will be raised.
        """
        if self._real_repository is not None:
            self._real_repository.refresh_data()
        self._unstacked_provider.disable_cache()
        self._unstacked_provider.enable_cache()

    def revision_ids_to_search_result(self, result_set):
        """Convert a set of revision ids to a graph SearchResult."""
        result_parents = set()
        for parents in self.get_graph().get_parent_map(result_set).values():
            result_parents.update(parents)
        included_keys = result_set.intersection(result_parents)
        start_keys = result_set.difference(included_keys)
        exclude_keys = result_parents.difference(result_set)
        result = vf_search.SearchResult(start_keys, exclude_keys, len(result_set), result_set)
        return result

    def search_missing_revision_ids(self, other, find_ghosts=True, revision_ids=None, if_present_ids=None, limit=None):
        """Return the revision ids that other has that this does not.

        These are returned in topological order.

        revision_id: only return revision ids included by revision_id.
        """
        with self.lock_read():
            inter_repo = _mod_repository.InterRepository.get(other, self)
            return inter_repo.search_missing_revision_ids(find_ghosts=find_ghosts, revision_ids=revision_ids, if_present_ids=if_present_ids, limit=limit)

    def fetch(self, source, revision_id=None, find_ghosts=False, fetch_spec=None, lossy=False):
        if fetch_spec is not None and revision_id is not None:
            raise AssertionError('fetch_spec and revision_id are mutually exclusive.')
        if self.is_in_write_group():
            raise errors.InternalBzrError('May not fetch while in a write group.')
        if self.has_same_location(source) and fetch_spec is None and self._has_same_fallbacks(source):
            if revision_id is not None and (not _mod_revision.is_null(revision_id)):
                self.get_revision(revision_id)
            return _mod_repository.FetchResult(0)
        inter = _mod_repository.InterRepository.get(source, self)
        if fetch_spec is not None and (not getattr(inter, 'supports_fetch_spec', False)):
            raise errors.UnsupportedOperation('fetch_spec not supported for %r' % inter)
        return inter.fetch(revision_id=revision_id, find_ghosts=find_ghosts, fetch_spec=fetch_spec, lossy=lossy)

    def create_bundle(self, target, base, fileobj, format=None):
        self._ensure_real()
        self._real_repository.create_bundle(target, base, fileobj, format)

    def fileids_altered_by_revision_ids(self, revision_ids):
        self._ensure_real()
        return self._real_repository.fileids_altered_by_revision_ids(revision_ids)

    def _get_versioned_file_checker(self, revisions, revision_versions_cache):
        self._ensure_real()
        return self._real_repository._get_versioned_file_checker(revisions, revision_versions_cache)

    def _iter_files_bytes_rpc(self, desired_files, absent):
        path = self.controldir._path_for_remote_call(self._client)
        lines = []
        identifiers = []
        for file_id, revid, identifier in desired_files:
            lines.append(b''.join([file_id, b'\x00', revid]))
            identifiers.append(identifier)
        response_tuple, response_handler = self._call_with_body_bytes_expecting_body(b'Repository.iter_files_bytes', (path,), b'\n'.join(lines))
        if response_tuple != (b'ok',):
            response_handler.cancel_read_body()
            raise errors.UnexpectedSmartServerResponse(response_tuple)
        byte_stream = response_handler.read_streamed_body()

        def decompress_stream(start, byte_stream, unused):
            decompressor = zlib.decompressobj()
            yield decompressor.decompress(start)
            while decompressor.unused_data == b'':
                try:
                    data = next(byte_stream)
                except StopIteration:
                    break
                yield decompressor.decompress(data)
            yield decompressor.flush()
            unused.append(decompressor.unused_data)
        unused = b''
        while True:
            while b'\n' not in unused:
                try:
                    unused += next(byte_stream)
                except StopIteration:
                    return
            header, rest = unused.split(b'\n', 1)
            args = header.split(b'\x00')
            if args[0] == b'absent':
                absent[identifiers[int(args[3])]] = (args[1], args[2])
                unused = rest
                continue
            elif args[0] == b'ok':
                idx = int(args[1])
            else:
                raise errors.UnexpectedSmartServerResponse(args)
            unused_chunks = []
            yield (identifiers[idx], decompress_stream(rest, byte_stream, unused_chunks))
            unused = b''.join(unused_chunks)

    def iter_files_bytes(self, desired_files):
        """See Repository.iter_file_bytes.
        """
        try:
            absent = {}
            for identifier, bytes_iterator in self._iter_files_bytes_rpc(desired_files, absent):
                yield (identifier, bytes_iterator)
            for fallback in self._fallback_repositories:
                if not absent:
                    break
                desired_files = [(key[0], key[1], identifier) for identifier, key in absent.items()]
                for identifier, bytes_iterator in fallback.iter_files_bytes(desired_files):
                    del absent[identifier]
                    yield (identifier, bytes_iterator)
            if absent:
                missing_identifier = next(iter(absent))
                missing_key = absent[missing_identifier]
                raise errors.RevisionNotPresent(revision_id=missing_key[1], file_id=missing_key[0])
        except errors.UnknownSmartMethod:
            self._ensure_real()
            for identifier, bytes_iterator in self._real_repository.iter_files_bytes(desired_files):
                yield (identifier, bytes_iterator)

    def get_cached_parent_map(self, revision_ids):
        """See breezy.CachingParentsProvider.get_cached_parent_map"""
        return self._unstacked_provider.get_cached_parent_map(revision_ids)

    def get_parent_map(self, revision_ids):
        """See breezy.Graph.get_parent_map()."""
        return self._make_parents_provider().get_parent_map(revision_ids)

    def _get_parent_map_rpc(self, keys):
        """Helper for get_parent_map that performs the RPC."""
        medium = self._client._medium
        if medium._is_remote_before((1, 2)):
            rg = self._get_revision_graph(None)
            for node_id, parent_ids in rg.items():
                if parent_ids == ():
                    rg[node_id] = (NULL_REVISION,)
            rg[NULL_REVISION] = ()
            return rg
        keys = set(keys)
        if None in keys:
            raise ValueError('get_parent_map(None) is not valid')
        if NULL_REVISION in keys:
            keys.discard(NULL_REVISION)
            found_parents = {NULL_REVISION: ()}
            if not keys:
                return found_parents
        else:
            found_parents = {}
        parents_map = self._unstacked_provider.get_cached_map()
        if parents_map is None:
            parents_map = {}
        if _DEFAULT_SEARCH_DEPTH <= 0:
            start_set, stop_keys, key_count = vf_search.search_result_from_parent_map(parents_map, self._unstacked_provider.missing_keys)
        else:
            start_set, stop_keys, key_count = vf_search.limited_search_result_from_parent_map(parents_map, self._unstacked_provider.missing_keys, keys, depth=_DEFAULT_SEARCH_DEPTH)
        recipe = ('manual', start_set, stop_keys, key_count)
        body = self._serialise_search_recipe(recipe)
        path = self.controldir._path_for_remote_call(self._client)
        for key in keys:
            if not isinstance(key, bytes):
                raise ValueError('key {!r} not a bytes string'.format(key))
        verb = b'Repository.get_parent_map'
        args = (path, b'include-missing:') + tuple(keys)
        try:
            response = self._call_with_body_bytes_expecting_body(verb, args, body)
        except errors.UnknownSmartMethod:
            warning('Server is too old for fast get_parent_map, reconnecting.  (Upgrade the server to Bazaar 1.2 to avoid this)')
            medium.disconnect()
            medium._remember_remote_is_before((1, 2))
            return self._get_parent_map_rpc(keys)
        response_tuple, response_handler = response
        if response_tuple[0] not in [b'ok']:
            response_handler.cancel_read_body()
            raise errors.UnexpectedSmartServerResponse(response_tuple)
        if response_tuple[0] == b'ok':
            coded = bz2.decompress(response_handler.read_body_bytes())
            if coded == b'':
                return {}
            lines = coded.split(b'\n')
            revision_graph = {}
            for line in lines:
                d = tuple(line.split())
                if len(d) > 1:
                    revision_graph[d[0]] = d[1:]
                elif d[0].startswith(b'missing:'):
                    revid = d[0][8:]
                    self._unstacked_provider.note_missing_key(revid)
                else:
                    revision_graph[d[0]] = (NULL_REVISION,)
            return revision_graph

    def get_signature_text(self, revision_id):
        with self.lock_read():
            path = self.controldir._path_for_remote_call(self._client)
            try:
                response_tuple, response_handler = self._call_expecting_body(b'Repository.get_revision_signature_text', path, revision_id)
            except errors.UnknownSmartMethod:
                self._ensure_real()
                return self._real_repository.get_signature_text(revision_id)
            except errors.NoSuchRevision as err:
                for fallback in self._fallback_repositories:
                    try:
                        return fallback.get_signature_text(revision_id)
                    except errors.NoSuchRevision:
                        pass
                raise err
            else:
                if response_tuple[0] != b'ok':
                    raise errors.UnexpectedSmartServerResponse(response_tuple)
                return response_handler.read_body_bytes()

    def _get_inventory_xml(self, revision_id):
        with self.lock_read():
            self._ensure_real()
            return self._real_repository._get_inventory_xml(revision_id)

    def reconcile(self, other=None, thorough=False):
        from ..reconcile import ReconcileResult
        with self.lock_write():
            path = self.controldir._path_for_remote_call(self._client)
            try:
                response, handler = self._call_expecting_body(b'Repository.reconcile', path, self._lock_token)
            except (errors.UnknownSmartMethod, errors.TokenLockingNotSupported):
                self._ensure_real()
                return self._real_repository.reconcile(other=other, thorough=thorough)
            if response != (b'ok',):
                raise errors.UnexpectedSmartServerResponse(response)
            body = handler.read_body_bytes()
            result = ReconcileResult()
            result.garbage_inventories = None
            result.inconsistent_parents = None
            result.aborted = None
            for line in body.split(b'\n'):
                if not line:
                    continue
                key, val_text = line.split(b':')
                if key == b'garbage_inventories':
                    result.garbage_inventories = int(val_text)
                elif key == b'inconsistent_parents':
                    result.inconsistent_parents = int(val_text)
                else:
                    mutter('unknown reconcile key %r' % key)
            return result

    def all_revision_ids(self):
        path = self.controldir._path_for_remote_call(self._client)
        try:
            response_tuple, response_handler = self._call_expecting_body(b'Repository.all_revision_ids', path)
        except errors.UnknownSmartMethod:
            self._ensure_real()
            return self._real_repository.all_revision_ids()
        if response_tuple != (b'ok',):
            raise errors.UnexpectedSmartServerResponse(response_tuple)
        revids = set(response_handler.read_body_bytes().splitlines())
        for fallback in self._fallback_repositories:
            revids.update(set(fallback.all_revision_ids()))
        return list(revids)

    def _filtered_revision_trees(self, revision_ids, file_ids):
        """Return Tree for a revision on this branch with only some files.

        :param revision_ids: a sequence of revision-ids;
          a revision-id may not be None or b'null:'
        :param file_ids: if not None, the result is filtered
          so that only those file-ids, their parents and their
          children are included.
        """
        inventories = self.iter_inventories(revision_ids)
        for inv in inventories:
            filtered_inv = inv.filter(file_ids)
            yield InventoryRevisionTree(self, filtered_inv, filtered_inv.revision_id)

    def get_revision_delta(self, revision_id):
        with self.lock_read():
            r = self.get_revision(revision_id)
            return list(self.get_revision_deltas([r]))[0]

    def revision_trees(self, revision_ids):
        with self.lock_read():
            inventories = self.iter_inventories(revision_ids)
            for inv in inventories:
                yield RemoteInventoryTree(self, inv, inv.revision_id)

    def get_revision_reconcile(self, revision_id):
        with self.lock_read():
            self._ensure_real()
            return self._real_repository.get_revision_reconcile(revision_id)

    def check(self, revision_ids=None, callback_refs=None, check_repo=True):
        with self.lock_read():
            self._ensure_real()
            return self._real_repository.check(revision_ids=revision_ids, callback_refs=callback_refs, check_repo=check_repo)

    def copy_content_into(self, destination, revision_id=None):
        """Make a complete copy of the content in self into destination.

        This is a destructive operation! Do not use it on existing
        repositories.
        """
        interrepo = _mod_repository.InterRepository.get(self, destination)
        return interrepo.copy_content(revision_id)

    def _copy_repository_tarball(self, to_bzrdir, revision_id=None):
        import tarfile
        note(gettext('Copying repository content as tarball...'))
        tar_file = self._get_tarball('bz2')
        if tar_file is None:
            return None
        destination = to_bzrdir.create_repository()
        with tarfile.open('repository', fileobj=tar_file, mode='r|bz2') as tar, osutils.TemporaryDirectory() as tmpdir:
            tar.extractall(tmpdir)
            tmp_bzrdir = _mod_bzrdir.BzrDir.open(tmpdir)
            tmp_repo = tmp_bzrdir.open_repository()
            tmp_repo.copy_content_into(destination, revision_id)
        return destination

    @property
    def inventories(self):
        """Decorate the real repository for now.

        In the long term a full blown network facility is needed to
        avoid creating a real repository object locally.
        """
        self._ensure_real()
        return self._real_repository.inventories

    def pack(self, hint=None, clean_obsolete_packs=False):
        """Compress the data within the repository.
        """
        if hint is None:
            body = b''
        else:
            body = b''.join([l.encode('ascii') + b'\n' for l in hint])
        with self.lock_write():
            path = self.controldir._path_for_remote_call(self._client)
            try:
                response, handler = self._call_with_body_bytes_expecting_body(b'Repository.pack', (path, self._lock_token, str(clean_obsolete_packs).encode('ascii')), body)
            except errors.UnknownSmartMethod:
                self._ensure_real()
                return self._real_repository.pack(hint=hint, clean_obsolete_packs=clean_obsolete_packs)
            handler.cancel_read_body()
            if response != (b'ok',):
                raise errors.UnexpectedSmartServerResponse(response)

    @property
    def revisions(self):
        """Decorate the real repository for now.

        In the long term a full blown network facility is needed.
        """
        self._ensure_real()
        return self._real_repository.revisions

    def set_make_working_trees(self, new_value):
        if new_value:
            new_value_str = b'True'
        else:
            new_value_str = b'False'
        path = self.controldir._path_for_remote_call(self._client)
        try:
            response = self._call(b'Repository.set_make_working_trees', path, new_value_str)
        except errors.UnknownSmartMethod:
            self._ensure_real()
            self._real_repository.set_make_working_trees(new_value)
        else:
            if response[0] != b'ok':
                raise errors.UnexpectedSmartServerResponse(response)

    @property
    def signatures(self):
        """Decorate the real repository for now.

        In the long term a full blown network facility is needed to avoid
        creating a real repository object locally.
        """
        self._ensure_real()
        return self._real_repository.signatures

    def sign_revision(self, revision_id, gpg_strategy):
        with self.lock_write():
            testament = _mod_testament.Testament.from_revision(self, revision_id)
            plaintext = testament.as_short_text()
            self.store_revision_signature(gpg_strategy, plaintext, revision_id)

    @property
    def texts(self):
        """Decorate the real repository for now.

        In the long term a full blown network facility is needed to avoid
        creating a real repository object locally.
        """
        self._ensure_real()
        return self._real_repository.texts

    def _iter_revisions_rpc(self, revision_ids):
        body = b'\n'.join(revision_ids)
        path = self.controldir._path_for_remote_call(self._client)
        response_tuple, response_handler = self._call_with_body_bytes_expecting_body(b'Repository.iter_revisions', (path,), body)
        if response_tuple[0] != b'ok':
            raise errors.UnexpectedSmartServerResponse(response_tuple)
        serializer_format = response_tuple[1].decode('ascii')
        serializer = serializer_format_registry.get(serializer_format)
        byte_stream = response_handler.read_streamed_body()
        decompressor = zlib.decompressobj()
        chunks = []
        for bytes in byte_stream:
            chunks.append(decompressor.decompress(bytes))
            if decompressor.unused_data != b'':
                chunks.append(decompressor.flush())
                yield serializer.read_revision_from_string(b''.join(chunks))
                unused = decompressor.unused_data
                decompressor = zlib.decompressobj()
                chunks = [decompressor.decompress(unused)]
        chunks.append(decompressor.flush())
        text = b''.join(chunks)
        if text != b'':
            yield serializer.read_revision_from_string(b''.join(chunks))

    def iter_revisions(self, revision_ids):
        for rev_id in revision_ids:
            if not rev_id or not isinstance(rev_id, bytes):
                raise errors.InvalidRevisionId(revision_id=rev_id, branch=self)
        with self.lock_read():
            try:
                missing = set(revision_ids)
                for rev in self._iter_revisions_rpc(revision_ids):
                    missing.remove(rev.revision_id)
                    yield (rev.revision_id, rev)
                for fallback in self._fallback_repositories:
                    if not missing:
                        break
                    for revid, rev in fallback.iter_revisions(missing):
                        if rev is not None:
                            yield (revid, rev)
                            missing.remove(revid)
                for revid in missing:
                    yield (revid, None)
            except errors.UnknownSmartMethod:
                self._ensure_real()
                yield from self._real_repository.iter_revisions(revision_ids)

    def supports_rich_root(self):
        return self._format.rich_root_data

    @property
    def _serializer(self):
        return self._format._serializer

    def store_revision_signature(self, gpg_strategy, plaintext, revision_id):
        with self.lock_write():
            signature = gpg_strategy.sign(plaintext, gpg.MODE_CLEAR)
            self.add_signature_text(revision_id, signature)

    def add_signature_text(self, revision_id, signature):
        if self._real_repository:
            self._ensure_real()
            return self._real_repository.add_signature_text(revision_id, signature)
        path = self.controldir._path_for_remote_call(self._client)
        response, handler = self._call_with_body_bytes_expecting_body(b'Repository.add_signature_text', (path, self._lock_token, revision_id) + tuple([token.encode('utf-8') for token in self._write_group_tokens]), signature)
        handler.cancel_read_body()
        self.refresh_data()
        if response[0] != b'ok':
            raise errors.UnexpectedSmartServerResponse(response)
        self._write_group_tokens = [token.decode('utf-8') for token in response[1:]]

    def has_signature_for_revision_id(self, revision_id):
        path = self.controldir._path_for_remote_call(self._client)
        try:
            response = self._call(b'Repository.has_signature_for_revision_id', path, revision_id)
        except errors.UnknownSmartMethod:
            self._ensure_real()
            return self._real_repository.has_signature_for_revision_id(revision_id)
        if response[0] not in (b'yes', b'no'):
            raise SmartProtocolError('unexpected response code {}'.format(response))
        if response[0] == b'yes':
            return True
        for fallback in self._fallback_repositories:
            if fallback.has_signature_for_revision_id(revision_id):
                return True
        return False

    def verify_revision_signature(self, revision_id, gpg_strategy):
        with self.lock_read():
            if not self.has_signature_for_revision_id(revision_id):
                return (gpg.SIGNATURE_NOT_SIGNED, None)
            signature = self.get_signature_text(revision_id)
            testament = _mod_testament.Testament.from_revision(self, revision_id)
            status, key, signed_plaintext = gpg_strategy.verify(signature)
            if testament.as_short_text() != signed_plaintext:
                return (gpg.SIGNATURE_NOT_VALID, None)
            return (status, key)

    def item_keys_introduced_by(self, revision_ids, _files_pb=None):
        self._ensure_real()
        return self._real_repository.item_keys_introduced_by(revision_ids, _files_pb=_files_pb)

    def _find_inconsistent_revision_parents(self, revisions_iterator=None):
        self._ensure_real()
        return self._real_repository._find_inconsistent_revision_parents(revisions_iterator)

    def _check_for_inconsistent_revision_parents(self):
        self._ensure_real()
        return self._real_repository._check_for_inconsistent_revision_parents()

    def _make_parents_provider(self, other=None):
        providers = [self._unstacked_provider]
        if other is not None:
            providers.insert(0, other)
        return graph.StackedParentsProvider(_LazyListJoin(providers, self._fallback_repositories))

    def _serialise_search_recipe(self, recipe):
        """Serialise a graph search recipe.

        :param recipe: A search recipe (start, stop, count).
        :return: Serialised bytes.
        """
        start_keys = b' '.join(recipe[1])
        stop_keys = b' '.join(recipe[2])
        count = str(recipe[3]).encode('ascii')
        return b'\n'.join((start_keys, stop_keys, count))

    def _serialise_search_result(self, search_result):
        parts = search_result.get_network_struct()
        return b'\n'.join(parts)

    def autopack(self):
        path = self.controldir._path_for_remote_call(self._client)
        try:
            response = self._call(b'PackRepository.autopack', path)
        except errors.UnknownSmartMethod:
            self._ensure_real()
            self._real_repository._pack_collection.autopack()
            return
        self.refresh_data()
        if response[0] != b'ok':
            raise errors.UnexpectedSmartServerResponse(response)

    def _revision_archive(self, revision_id, format, name, root, subdir, force_mtime=None):
        path = self.controldir._path_for_remote_call(self._client)
        format = format or ''
        root = root or ''
        subdir = subdir or ''
        force_mtime = int(force_mtime) if force_mtime is not None else None
        try:
            response, protocol = self._call_expecting_body(b'Repository.revision_archive', path, revision_id, format.encode('ascii'), os.path.basename(name).encode('utf-8'), root.encode('utf-8'), subdir.encode('utf-8'), force_mtime)
        except errors.UnknownSmartMethod:
            return None
        if response[0] == b'ok':
            return iter([protocol.read_body_bytes()])
        raise errors.UnexpectedSmartServerResponse(response)

    def _annotate_file_revision(self, revid, tree_path, file_id, default_revision):
        path = self.controldir._path_for_remote_call(self._client)
        tree_path = tree_path.encode('utf-8')
        file_id = file_id or b''
        default_revision = default_revision or b''
        try:
            response, handler = self._call_expecting_body(b'Repository.annotate_file_revision', path, revid, tree_path, file_id, default_revision)
        except errors.UnknownSmartMethod:
            return None
        if response[0] != b'ok':
            raise errors.UnexpectedSmartServerResponse(response)
        return map(tuple, bencode.bdecode(handler.read_body_bytes()))