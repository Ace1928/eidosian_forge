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
class RemoteBranch(branch.Branch, _RpcHelper, lock._RelockDebugMixin):
    """Branch stored on a server accessed by HPSS RPC.

    At the moment most operations are mapped down to simple file operations.
    """
    _real_branch: Optional['bzrbranch.BzrBranch']
    _format: RemoteBranchFormat
    repository: RemoteRepository

    @property
    def control_transport(self) -> _mod_transport.Transport:
        return self._transport

    def __init__(self, remote_bzrdir: RemoteBzrDir, remote_repository: RemoteRepository, real_branch: Optional['bzrbranch.BzrBranch']=None, _client=None, format=None, setup_stacking: bool=True, name: Optional[str]=None, possible_transports: Optional[List[_mod_transport.Transport]]=None):
        """Create a RemoteBranch instance.

        :param real_branch: An optional local implementation of the branch
            format, usually accessing the data via the VFS.
        :param _client: Private parameter for testing.
        :param format: A RemoteBranchFormat object, None to create one
            automatically. If supplied it should have a network_name already
            supplied.
        :param setup_stacking: If True make an RPC call to determine the
            stacked (or not) status of the branch. If False assume the branch
            is not stacked.
        :param name: Colocated branch name
        """
        self.controldir = remote_bzrdir
        self.name = name
        if _client is not None:
            self._client = _client
        else:
            self._client = remote_bzrdir._client
        self.repository = remote_repository
        if real_branch is not None:
            self._real_branch = real_branch
            real_repo: _mod_repository.Repository = real_branch.repository
            if isinstance(real_repo, RemoteRepository):
                real_repo._ensure_real()
                real_repo = real_repo._real_repository
            self.repository._set_real_repository(real_repo)
            real_branch.repository = self.repository
        else:
            self._real_branch = None
        self._clear_cached_state()
        self.base = self.controldir.user_url
        self._name = name
        self._control_files = None
        self._lock_mode = None
        self._lock_token = None
        self._repo_lock_token = None
        self._lock_count = 0
        self._leave_lock = False
        self.conf_store = None
        if format is None:
            self._format = RemoteBranchFormat()
            if self._real_branch is not None:
                self._format._network_name = self._real_branch._format.network_name()
        else:
            self._format = format
        self._real_ignore_fallbacks = not setup_stacking
        if not self._format._network_name:
            self._ensure_real()
            if not self._real_branch:
                raise AssertionError
            self._format._network_name = self._real_branch._format.network_name()
        self.tags = self._format.make_tags(self)
        hooks = branch.Branch.hooks['open']
        for hook in hooks:
            hook(self)
        self._is_stacked = False
        if setup_stacking:
            self._setup_stacking(possible_transports)

    def _setup_stacking(self, possible_transports):
        try:
            fallback_url = self.get_stacked_on_url()
        except (errors.NotStacked, branch.UnstackableBranchFormat, errors.UnstackableRepositoryFormat) as e:
            return
        self._is_stacked = True
        if possible_transports is None:
            possible_transports = []
        else:
            possible_transports = list(possible_transports)
        possible_transports.append(self.controldir.root_transport)
        self._activate_fallback_location(fallback_url, possible_transports=possible_transports)

    def _get_config(self):
        return RemoteBranchConfig(self)

    def _get_config_store(self):
        if self.conf_store is None:
            self.conf_store = RemoteBranchStore(self)
        return self.conf_store

    def store_uncommitted(self, creator):
        self._ensure_real()
        if self._real_branch is None:
            raise AssertionError
        return self._real_branch.store_uncommitted(creator)

    def get_unshelver(self, tree):
        self._ensure_real()
        if self._real_branch is None:
            raise AssertionError
        return self._real_branch.get_unshelver(tree)

    def _get_real_transport(self) -> _mod_transport.Transport:
        self._ensure_real()
        if self._real_branch is None:
            raise AssertionError
        return self._real_branch._transport
    _transport = property(_get_real_transport)

    def __str__(self):
        return '{}({})'.format(self.__class__.__name__, self.base)
    __repr__ = __str__

    def _ensure_real(self):
        """Ensure that there is a _real_branch set.

        Used before calls to self._real_branch.
        """
        if self._real_branch is None:
            if not vfs.vfs_enabled():
                raise AssertionError('smart server vfs must be enabled to use vfs implementation')
            self.controldir._ensure_real()
            self._real_branch = self.controldir._real_bzrdir.open_branch(ignore_fallbacks=self._real_ignore_fallbacks, name=self._name)
            self._real_branch.conf_store = self.conf_store
            if self.repository._real_repository is None:
                real_repo = self._real_branch.repository
                if isinstance(real_repo, RemoteRepository):
                    real_repo._ensure_real()
                    real_repo = real_repo._real_repository
                self.repository._set_real_repository(real_repo)
            self._real_branch.repository = self.repository
            if self._lock_mode == 'r':
                self._real_branch.lock_read()
            elif self._lock_mode == 'w':
                self._real_branch.lock_write(token=self._lock_token)

    def _translate_error(self, err, **context):
        self.repository._translate_error(err, branch=self, **context)

    def _clear_cached_state(self):
        super()._clear_cached_state()
        self._tags_bytes = None
        if self._real_branch is not None:
            self._real_branch._clear_cached_state()

    def _clear_cached_state_of_remote_branch_only(self):
        """Like _clear_cached_state, but doesn't clear the cache of
        self._real_branch.

        This is useful when falling back to calling a method of
        self._real_branch that changes state.  In that case the underlying
        branch changes, so we need to invalidate this RemoteBranch's cache of
        it.  However, there's no need to invalidate the _real_branch's cache
        too, in fact doing so might harm performance.
        """
        super()._clear_cached_state()

    @property
    def control_files(self):
        if self._control_files is None:
            self._control_files = RemoteBranchLockableFiles(self.controldir, self._client)
        return self._control_files

    def get_physical_lock_status(self):
        """See Branch.get_physical_lock_status()."""
        try:
            response = self._client.call(b'Branch.get_physical_lock_status', self._remote_path())
        except errors.UnknownSmartMethod:
            self._ensure_real()
            return self._real_branch.get_physical_lock_status()
        if response[0] not in (b'yes', b'no'):
            raise errors.UnexpectedSmartServerResponse(response)
        return response[0] == b'yes'

    def get_stacked_on_url(self):
        """Get the URL this branch is stacked against.

        :raises NotStacked: If the branch is not stacked.
        :raises UnstackableBranchFormat: If the branch does not support
            stacking.
        :raises UnstackableRepositoryFormat: If the repository does not support
            stacking.
        """
        try:
            response = self._client.call(b'Branch.get_stacked_on_url', self._remote_path())
        except errors.ErrorFromSmartServer as err:
            _translate_error(err, branch=self)
        except errors.UnknownSmartMethod as err:
            self._ensure_real()
            return self._real_branch.get_stacked_on_url()
        if response[0] != b'ok':
            raise errors.UnexpectedSmartServerResponse(response)
        return response[1].decode('utf-8')

    def _check_stackable_repo(self) -> None:
        if not self.repository._format.supports_external_lookups:
            raise errors.UnstackableRepositoryFormat(self.repository._format, self.repository.user_url)

    def _unstack(self):
        """Change a branch to be unstacked, copying data as needed.

        Don't call this directly, use set_stacked_on_url(None).
        """
        with ui.ui_factory.nested_progress_bar() as pb:
            old_repository = self.repository
            if len(old_repository._fallback_repositories) != 1:
                raise AssertionError("can't cope with fallback repositories of %r (fallbacks: %r)" % (old_repository, old_repository._fallback_repositories))
            new_bzrdir = controldir.ControlDir.open(self.controldir.root_transport.base)
            new_repository = new_bzrdir.find_repository()
            if new_repository._fallback_repositories:
                raise AssertionError("didn't expect %r to have fallback_repositories" % (self.repository,))
            lock_token = old_repository.lock_write().repository_token
            self.repository = new_repository
            if self._real_branch is not None:
                self._real_branch.repository = new_repository
            self.repository.lock_write(token=lock_token)
            if lock_token is not None:
                old_repository.leave_lock_in_place()
            old_repository.unlock()
            if lock_token is not None:
                self.repository.dont_leave_lock_in_place()
            old_lock_count = 0
            while True:
                try:
                    old_repository.unlock()
                except errors.LockNotHeld:
                    break
                old_lock_count += 1
            if old_lock_count == 0:
                raise AssertionError('old_repository should have been locked at least once.')
            for i in range(old_lock_count - 1):
                self.repository.lock_write()
            with old_repository.lock_read():
                try:
                    tags_to_fetch = set(self.tags.get_reverse_tag_dict())
                except errors.TagsNotSupported:
                    tags_to_fetch = set()
                fetch_spec = vf_search.NotInOtherForRevs(self.repository, old_repository, required_ids=[self.last_revision()], if_present_ids=tags_to_fetch, find_ghosts=True).execute()
                self.repository.fetch(old_repository, fetch_spec=fetch_spec)

    def set_stacked_on_url(self, url):
        if not self._format.supports_stacking():
            raise UnstackableBranchFormat(self._format, self.user_url)
        with self.lock_write():
            self._check_stackable_repo()
            if not url:
                try:
                    self.get_stacked_on_url()
                except (errors.NotStacked, UnstackableBranchFormat, errors.UnstackableRepositoryFormat):
                    return
                self._unstack()
            else:
                self._activate_fallback_location(url, possible_transports=[self.controldir.root_transport])
            self._set_config_location('stacked_on_location', url)
        self.conf_store.save_changes()
        if not url:
            self._is_stacked = False
        else:
            self._is_stacked = True

    def _vfs_get_tags_bytes(self):
        self._ensure_real()
        return self._real_branch._get_tags_bytes()

    def _get_tags_bytes(self):
        with self.lock_read():
            if self._tags_bytes is None:
                self._tags_bytes = self._get_tags_bytes_via_hpss()
            return self._tags_bytes

    def _get_tags_bytes_via_hpss(self):
        medium = self._client._medium
        if medium._is_remote_before((1, 13)):
            return self._vfs_get_tags_bytes()
        try:
            response = self._call(b'Branch.get_tags_bytes', self._remote_path())
        except errors.UnknownSmartMethod:
            medium._remember_remote_is_before((1, 13))
            return self._vfs_get_tags_bytes()
        return response[0]

    def _vfs_set_tags_bytes(self, bytes):
        self._ensure_real()
        return self._real_branch._set_tags_bytes(bytes)

    def _set_tags_bytes(self, bytes):
        if self.is_locked():
            self._tags_bytes = bytes
        medium = self._client._medium
        if medium._is_remote_before((1, 18)):
            self._vfs_set_tags_bytes(bytes)
            return
        try:
            args = (self._remote_path(), self._lock_token, self._repo_lock_token)
            response = self._call_with_body_bytes(b'Branch.set_tags_bytes', args, bytes)
        except errors.UnknownSmartMethod:
            medium._remember_remote_is_before((1, 18))
            self._vfs_set_tags_bytes(bytes)

    def lock_read(self):
        """Lock the branch for read operations.

        :return: A breezy.lock.LogicalLockResult.
        """
        self.repository.lock_read()
        if not self._lock_mode:
            self._note_lock('r')
            self._lock_mode = 'r'
            self._lock_count = 1
            if self._real_branch is not None:
                self._real_branch.lock_read()
        else:
            self._lock_count += 1
        return lock.LogicalLockResult(self.unlock)

    def _remote_lock_write(self, token):
        if token is None:
            branch_token = repo_token = b''
        else:
            branch_token = token
            repo_token = self.repository.lock_write().repository_token
            self.repository.unlock()
        err_context = {'token': token}
        try:
            response = self._call(b'Branch.lock_write', self._remote_path(), branch_token, repo_token or b'', **err_context)
        except errors.LockContention as e:
            raise errors.LockContention('(remote lock)', self.repository.base.split('.bzr/')[0])
        if response[0] != b'ok':
            raise errors.UnexpectedSmartServerResponse(response)
        ok, branch_token, repo_token = response
        return (branch_token, repo_token)

    def lock_write(self, token=None):
        if not self._lock_mode:
            self._note_lock('w')
            remote_tokens = self._remote_lock_write(token)
            self._lock_token, self._repo_lock_token = remote_tokens
            if not self._lock_token:
                raise SmartProtocolError('Remote server did not return a token!')
            self.repository.lock_write(self._repo_lock_token, _skip_rpc=True)
            if self._real_branch is not None:
                self._real_branch.lock_write(token=self._lock_token)
            if token is not None:
                self._leave_lock = True
            else:
                self._leave_lock = False
            self._lock_mode = 'w'
            self._lock_count = 1
        elif self._lock_mode == 'r':
            raise errors.ReadOnlyError(self)
        else:
            if token is not None:
                if token != self._lock_token:
                    raise errors.TokenMismatch(token, self._lock_token)
            self._lock_count += 1
            self.repository.lock_write(self._repo_lock_token)
        return BranchWriteLockResult(self.unlock, self._lock_token or None)

    def _unlock(self, branch_token, repo_token):
        err_context = {'token': str((branch_token, repo_token))}
        response = self._call(b'Branch.unlock', self._remote_path(), branch_token, repo_token or b'', **err_context)
        if response == (b'ok',):
            return
        raise errors.UnexpectedSmartServerResponse(response)

    @only_raises(errors.LockNotHeld, errors.LockBroken)
    def unlock(self):
        try:
            self._lock_count -= 1
            if not self._lock_count:
                if self.conf_store is not None:
                    self.conf_store.save_changes()
                self._clear_cached_state()
                mode = self._lock_mode
                self._lock_mode = None
                if self._real_branch is not None:
                    if not self._leave_lock and mode == 'w' and self._repo_lock_token:
                        self._real_branch.repository.leave_lock_in_place()
                    self._real_branch.unlock()
                if mode != 'w':
                    return
                if not self._lock_token:
                    raise AssertionError('Locked, but no token!')
                branch_token = self._lock_token
                repo_token = self._repo_lock_token
                self._lock_token = None
                self._repo_lock_token = None
                if not self._leave_lock:
                    self._unlock(branch_token, repo_token)
        finally:
            self.repository.unlock()

    def break_lock(self):
        try:
            response = self._call(b'Branch.break_lock', self._remote_path())
        except errors.UnknownSmartMethod:
            self._ensure_real()
            return self._real_branch.break_lock()
        if response != (b'ok',):
            raise errors.UnexpectedSmartServerResponse(response)

    def leave_lock_in_place(self):
        if not self._lock_token:
            raise NotImplementedError(self.leave_lock_in_place)
        self._leave_lock = True

    def dont_leave_lock_in_place(self):
        if not self._lock_token:
            raise NotImplementedError(self.dont_leave_lock_in_place)
        self._leave_lock = False

    def get_rev_id(self, revno, history=None):
        if revno == 0:
            return _mod_revision.NULL_REVISION
        with self.lock_read():
            last_revision_info = self.last_revision_info()
            if revno < 0:
                raise errors.RevnoOutOfBounds(revno, (0, last_revision_info[0]))
            ok, result = self.repository.get_rev_id_for_revno(revno, last_revision_info)
            if ok:
                return result
            missing_parent = result[1]
            parent_map = self.repository.get_parent_map([missing_parent])
            if missing_parent in parent_map:
                missing_parent = parent_map[missing_parent]
            raise errors.NoSuchRevision(self, missing_parent)

    def _read_last_revision_info(self):
        response = self._call(b'Branch.last_revision_info', self._remote_path())
        if response[0] != b'ok':
            raise SmartProtocolError('unexpected response code {}'.format(response))
        revno = int(response[1])
        last_revision = response[2]
        return (revno, last_revision)

    def _gen_revision_history(self):
        """See Branch._gen_revision_history()."""
        if self._is_stacked:
            self._ensure_real()
            return self._real_branch._gen_revision_history()
        response_tuple, response_handler = self._call_expecting_body(b'Branch.revision_history', self._remote_path())
        if response_tuple[0] != b'ok':
            raise errors.UnexpectedSmartServerResponse(response_tuple)
        result = response_handler.read_body_bytes().split(b'\x00')
        if result == ['']:
            return []
        return result

    def _remote_path(self):
        return self.controldir._path_for_remote_call(self._client)

    def _set_last_revision_descendant(self, revision_id, other_branch, allow_diverged=False, allow_overwrite_descendant=False):
        old_revno, old_revid = self.last_revision_info()
        history = self._lefthand_history(revision_id)
        self._run_pre_change_branch_tip_hooks(len(history), revision_id)
        err_context = {'other_branch': other_branch}
        response = self._call(b'Branch.set_last_revision_ex', self._remote_path(), self._lock_token, self._repo_lock_token, revision_id, int(allow_diverged), int(allow_overwrite_descendant), **err_context)
        self._clear_cached_state()
        if len(response) != 3 and response[0] != b'ok':
            raise errors.UnexpectedSmartServerResponse(response)
        new_revno, new_revision_id = response[1:]
        self._last_revision_info_cache = (new_revno, new_revision_id)
        self._run_post_change_branch_tip_hooks(old_revno, old_revid)
        if self._real_branch is not None:
            cache = (new_revno, new_revision_id)
            self._real_branch._last_revision_info_cache = cache

    def _set_last_revision(self, revision_id):
        old_revno, old_revid = self.last_revision_info()
        history = self._lefthand_history(revision_id)
        self._run_pre_change_branch_tip_hooks(len(history), revision_id)
        self._clear_cached_state()
        response = self._call(b'Branch.set_last_revision', self._remote_path(), self._lock_token, self._repo_lock_token, revision_id)
        if response != (b'ok',):
            raise errors.UnexpectedSmartServerResponse(response)
        self._run_post_change_branch_tip_hooks(old_revno, old_revid)

    def _get_parent_location(self):
        medium = self._client._medium
        if medium._is_remote_before((1, 13)):
            return self._vfs_get_parent_location()
        try:
            response = self._call(b'Branch.get_parent', self._remote_path())
        except errors.UnknownSmartMethod:
            medium._remember_remote_is_before((1, 13))
            return self._vfs_get_parent_location()
        if len(response) != 1:
            raise errors.UnexpectedSmartServerResponse(response)
        parent_location = response[0]
        if parent_location == b'':
            return None
        return parent_location.decode('utf-8')

    def _vfs_get_parent_location(self):
        self._ensure_real()
        return self._real_branch._get_parent_location()

    def _set_parent_location(self, url):
        medium = self._client._medium
        if medium._is_remote_before((1, 15)):
            return self._vfs_set_parent_location(url)
        try:
            call_url = url or ''
            if isinstance(call_url, str):
                call_url = call_url.encode('utf-8')
            response = self._call(b'Branch.set_parent_location', self._remote_path(), self._lock_token, self._repo_lock_token, call_url)
        except errors.UnknownSmartMethod:
            medium._remember_remote_is_before((1, 15))
            return self._vfs_set_parent_location(url)
        if response != ():
            raise errors.UnexpectedSmartServerResponse(response)

    def _vfs_set_parent_location(self, url):
        self._ensure_real()
        return self._real_branch._set_parent_location(url)

    def pull(self, source, overwrite=False, stop_revision=None, **kwargs):
        with self.lock_write():
            self._clear_cached_state_of_remote_branch_only()
            self._ensure_real()
            return self._real_branch.pull(source, overwrite=overwrite, stop_revision=stop_revision, _override_hook_target=self, **kwargs)

    def push(self, target, overwrite=False, stop_revision=None, lossy=False, tag_selector=None):
        with self.lock_read():
            self._ensure_real()
            return self._real_branch.push(target, overwrite=overwrite, stop_revision=stop_revision, lossy=lossy, _override_hook_source_branch=self, tag_selector=tag_selector)

    def peek_lock_mode(self):
        return self._lock_mode

    def is_locked(self):
        return self._lock_count >= 1

    def revision_id_to_dotted_revno(self, revision_id):
        """Given a revision id, return its dotted revno.

        :return: a tuple like (1,) or (400,1,3).
        """
        with self.lock_read():
            try:
                response = self._call(b'Branch.revision_id_to_revno', self._remote_path(), revision_id)
            except errors.UnknownSmartMethod:
                self._ensure_real()
                return self._real_branch.revision_id_to_dotted_revno(revision_id)
            except UnknownErrorFromSmartServer as e:
                if e.error_tuple[1] == b'GhostRevisionsHaveNoRevno':
                    revid, ghost_revid = re.findall(b'{([^}]+)}', e.error_tuple[2])
                    raise errors.GhostRevisionsHaveNoRevno(revid, ghost_revid)
                raise
            if response[0] == b'ok':
                return tuple([int(x) for x in response[1:]])
            else:
                raise errors.UnexpectedSmartServerResponse(response)

    def revision_id_to_revno(self, revision_id):
        """Given a revision id on the branch mainline, return its revno.

        :return: an integer
        """
        with self.lock_read():
            try:
                response = self._call(b'Branch.revision_id_to_revno', self._remote_path(), revision_id)
            except errors.UnknownSmartMethod:
                self._ensure_real()
                return self._real_branch.revision_id_to_revno(revision_id)
            if response[0] == b'ok':
                if len(response) == 2:
                    return int(response[1])
                raise NoSuchRevision(self, revision_id)
            else:
                raise errors.UnexpectedSmartServerResponse(response)

    def set_last_revision_info(self, revno, revision_id):
        with self.lock_write():
            old_revno, old_revid = self.last_revision_info()
            self._run_pre_change_branch_tip_hooks(revno, revision_id)
            if not revision_id or not isinstance(revision_id, bytes):
                raise errors.InvalidRevisionId(revision_id=revision_id, branch=self)
            try:
                response = self._call(b'Branch.set_last_revision_info', self._remote_path(), self._lock_token, self._repo_lock_token, str(revno).encode('ascii'), revision_id)
            except errors.UnknownSmartMethod:
                self._ensure_real()
                self._clear_cached_state_of_remote_branch_only()
                self._real_branch.set_last_revision_info(revno, revision_id)
                self._last_revision_info_cache = (revno, revision_id)
                return
            if response == (b'ok',):
                self._clear_cached_state()
                self._last_revision_info_cache = (revno, revision_id)
                self._run_post_change_branch_tip_hooks(old_revno, old_revid)
                if self._real_branch is not None:
                    cache = self._last_revision_info_cache
                    self._real_branch._last_revision_info_cache = cache
            else:
                raise errors.UnexpectedSmartServerResponse(response)

    def generate_revision_history(self, revision_id, last_rev=None, other_branch=None):
        with self.lock_write():
            medium = self._client._medium
            if not medium._is_remote_before((1, 6)):
                try:
                    self._set_last_revision_descendant(revision_id, other_branch, allow_diverged=True, allow_overwrite_descendant=True)
                    return
                except errors.UnknownSmartMethod:
                    medium._remember_remote_is_before((1, 6))
            self._clear_cached_state_of_remote_branch_only()
            graph = self.repository.get_graph()
            last_revno, last_revid = self.last_revision_info()
            known_revision_ids = [(last_revid, last_revno), (_mod_revision.NULL_REVISION, 0)]
            if last_rev is not None:
                if not graph.is_ancestor(last_rev, revision_id):
                    raise errors.DivergedBranches(self, other_branch)
            revno = graph.find_distance_to_null(revision_id, known_revision_ids)
            self.set_last_revision_info(revno, revision_id)

    def set_push_location(self, location):
        self._set_config_location('push_location', location)

    def heads_to_fetch(self):
        if self._format._use_default_local_heads_to_fetch():
            return branch.Branch.heads_to_fetch(self)
        medium = self._client._medium
        if medium._is_remote_before((2, 4)):
            return self._vfs_heads_to_fetch()
        try:
            return self._rpc_heads_to_fetch()
        except errors.UnknownSmartMethod:
            medium._remember_remote_is_before((2, 4))
            return self._vfs_heads_to_fetch()

    def _rpc_heads_to_fetch(self):
        response = self._call(b'Branch.heads_to_fetch', self._remote_path())
        if len(response) != 2:
            raise errors.UnexpectedSmartServerResponse(response)
        must_fetch, if_present_fetch = response
        return (set(must_fetch), set(if_present_fetch))

    def _vfs_heads_to_fetch(self):
        self._ensure_real()
        return self._real_branch.heads_to_fetch()

    def reconcile(self, thorough=True):
        """Make sure the data stored in this branch is consistent."""
        from .reconcile import BranchReconciler
        with self.lock_write():
            reconciler = BranchReconciler(self, thorough=thorough)
            return reconciler.reconcile()

    def get_reference_info(self, file_id):
        """Get the tree_path and branch_location for a tree reference."""
        if not self._format.supports_reference_locations:
            raise errors.UnsupportedOperation(self.get_reference_info, self)
        return self._get_all_reference_info().get(file_id, (None, None))

    def set_reference_info(self, file_id, branch_location, tree_path=None):
        """Set the branch location to use for a tree reference."""
        if not self._format.supports_reference_locations:
            raise errors.UnsupportedOperation(self.set_reference_info, self)
        self._ensure_real()
        self._real_branch.set_reference_info(file_id, branch_location, tree_path)

    def _set_all_reference_info(self, reference_info):
        if not self._format.supports_reference_locations:
            raise errors.UnsupportedOperation(self.set_reference_info, self)
        self._ensure_real()
        self._real_branch._set_all_reference_info(reference_info)

    def _get_all_reference_info(self):
        if not self._format.supports_reference_locations:
            return {}
        try:
            response, handler = self._call_expecting_body(b'Branch.get_all_reference_info', self._remote_path())
        except errors.UnknownSmartMethod:
            self._ensure_real()
            return self._real_branch._get_all_reference_info()
        if len(response) and response[0] != b'ok':
            raise errors.UnexpectedSmartServerResponse(response)
        ret = {}
        for f, u, p in bencode.bdecode(handler.read_body_bytes()):
            ret[f] = (u.decode('utf-8'), p.decode('utf-8') if p else None)
        return ret

    def reference_parent(self, file_id, path, possible_transports=None):
        """Return the parent branch for a tree-reference.

        :param path: The path of the nested tree in the tree
        :return: A branch associated with the nested tree
        """
        branch_location = self.get_reference_info(file_id)[0]
        if branch_location is None:
            try:
                return branch.Branch.open_from_transport(self.controldir.root_transport.clone(path), possible_transports=possible_transports)
            except errors.NotBranchError:
                return None
        return branch.Branch.open(urlutils.join(urlutils.strip_segment_parameters(self.user_url), branch_location), possible_transports=possible_transports)