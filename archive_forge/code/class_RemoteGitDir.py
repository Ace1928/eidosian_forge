import gzip
import re
from dulwich.refs import SymrefLoop
from .. import config, debug, errors, osutils, trace, ui, urlutils
from ..controldir import BranchReferenceLoop
from ..errors import (AlreadyBranchError, BzrError, ConnectionReset,
from ..push import PushResult
from ..revision import NULL_REVISION
from ..revisiontree import RevisionTree
from ..transport import (NoSuchFile, Transport,
from . import is_github_url, lazy_check_versions, user_agent_for_github
import os
import select
import urllib.parse as urlparse
import dulwich
import dulwich.client
from dulwich.errors import GitProtocolError, HangupException
from dulwich.pack import (PACK_SPOOL_FILE_MAX_SIZE, Pack, load_pack_index,
from dulwich.protocol import ZERO_SHA
from dulwich.refs import SYMREF, DictRefsContainer
from dulwich.repo import NotGitRepository
from .branch import (GitBranch, GitBranchFormat, GitBranchPushResult, GitTags,
from .dir import GitControlDirFormat, GitDir
from .errors import GitSmartRemoteNotSupported
from .mapping import encode_git_path, mapping_registry
from .object_store import get_object_store
from .push import remote_divergence
from .refs import (branch_name_to_ref, is_peeled, ref_to_tag_name,
from .repository import GitRepository, GitRepositoryFormat
class RemoteGitDir(GitDir):

    def __init__(self, transport, format, client, client_path):
        self._format = format
        self.root_transport = transport
        self.transport = transport
        self._mode_check_done = None
        self._client = client
        self._client_path = client_path
        self.base = self.root_transport.base
        self._refs = None

    @property
    def _gitrepository_class(self):
        return RemoteGitRepository

    def archive(self, format, committish, write_data, progress=None, write_error=None, subdirs=None, prefix=None, recurse_nested=False):
        if recurse_nested:
            raise NotImplementedError('recurse_nested is not yet supported')
        if progress is None:
            pb = ui.ui_factory.nested_progress_bar()
            progress = DefaultProgressReporter(pb).progress
        else:
            pb = None

        def progress_wrapper(message):
            if message.startswith(b"fatal: Unknown archive format '"):
                format = message.strip()[len(b"fatal: Unknown archive format '"):-1]
                raise errors.NoSuchExportFormat(format.decode('ascii'))
            return progress(message)
        try:
            self._client.archive(self._client_path, committish, write_data, progress_wrapper, write_error, format=format.encode('ascii') if format else None, subdirs=subdirs, prefix=encode_git_path(prefix) if prefix else None)
        except HangupException as e:
            raise parse_git_hangup(self.transport.external_url(), e)
        except GitProtocolError as e:
            raise parse_git_error(self.transport.external_url(), e)
        finally:
            if pb is not None:
                pb.finished()

    def fetch_pack(self, determine_wants, graph_walker, pack_data, progress=None):
        if progress is None:
            pb = ui.ui_factory.nested_progress_bar()
            progress = DefaultProgressReporter(pb).progress
        else:
            pb = None
        try:
            result = self._client.fetch_pack(self._client_path, determine_wants, graph_walker, pack_data, progress)
            if result.refs is None:
                result.refs = {}
            self._refs = remote_refs_dict_to_container(result.refs, result.symrefs)
            return result
        except HangupException as e:
            raise parse_git_hangup(self.transport.external_url(), e)
        except GitProtocolError as e:
            raise parse_git_error(self.transport.external_url(), e)
        finally:
            if pb is not None:
                pb.finished()

    def send_pack(self, get_changed_refs, generate_pack_data, progress=None):
        if progress is None:
            pb = ui.ui_factory.nested_progress_bar()
            progress_reporter = DefaultProgressReporter(pb)
            progress = progress_reporter.progress
        else:
            progress_reporter = None
            pb = None

        def get_changed_refs_wrapper(remote_refs):
            if self._refs is not None:
                update_refs_container(self._refs, remote_refs)
            return get_changed_refs(remote_refs)
        try:
            result = self._client.send_pack(self._client_path, get_changed_refs_wrapper, generate_pack_data, progress)
            for ref, msg in list(result.ref_status.items()):
                if msg:
                    result.ref_status[ref] = RemoteGitError(msg=msg)
            if progress_reporter:
                for error in progress_reporter.errors:
                    m = _LOCK_REF_ERROR_MATCHER.match(error)
                    if m:
                        result.ref_status[m.group(1)] = LockContention(m.group(1), m.group(2))
            return result
        except HangupException as e:
            raise parse_git_hangup(self.transport.external_url(), e)
        except GitProtocolError as e:
            raise parse_git_error(self.transport.external_url(), e)
        finally:
            if pb is not None:
                pb.finished()

    def create_branch(self, name=None, repository=None, append_revisions_only=None, ref=None):
        refname = self._get_selected_ref(name, ref)
        if refname != b'HEAD' and refname in self.get_refs_container():
            raise AlreadyBranchError(self.user_url)
        ref_chain, sha = self.get_refs_container().follow(self._get_selected_ref(name))
        if ref_chain and ref_chain[0] == b'HEAD' and (len(ref_chain) > 1):
            refname = ref_chain[1]
        repo = self.open_repository()
        return RemoteGitBranch(self, repo, refname, sha)

    def destroy_branch(self, name=None):
        refname = self._get_selected_ref(name)

        def get_changed_refs(old_refs):
            ret = {}
            if refname not in old_refs:
                raise NotBranchError(self.user_url)
            ret[refname] = dulwich.client.ZERO_SHA
            return ret

        def generate_pack_data(have, want, ofs_delta=False, progress=None):
            return pack_objects_to_data([])
        result = self.send_pack(get_changed_refs, generate_pack_data)
        error = result.ref_status.get(refname)
        if error:
            raise error

    @property
    def user_url(self):
        return self.control_url

    @property
    def user_transport(self):
        return self.root_transport

    @property
    def control_url(self):
        return self.control_transport.base

    @property
    def control_transport(self):
        return self.root_transport

    def open_repository(self):
        return RemoteGitRepository(self)

    def get_branch_reference(self, name=None):
        ref = self._get_selected_ref(name)
        try:
            ref_chain, unused_sha = self.get_refs_container().follow(ref)
        except SymrefLoop:
            raise BranchReferenceLoop(self)
        if len(ref_chain) == 1:
            return None
        target_ref = ref_chain[1]
        from .refs import ref_to_branch_name
        try:
            branch_name = ref_to_branch_name(target_ref)
        except ValueError:
            params = {'ref': urlutils.quote(target_ref.decode('utf-8'), '')}
        else:
            if branch_name != '':
                params = {'branch': urlutils.quote(branch_name, '')}
            else:
                params = {}
        return urlutils.join_segment_parameters(self.user_url.rstrip('/'), params)

    def open_branch(self, name=None, unsupported=False, ignore_fallbacks=False, ref=None, possible_transports=None, nascent_ok=False):
        repo = self.open_repository()
        ref = self._get_selected_ref(name, ref)
        try:
            ref_chain, sha = self.get_refs_container().follow(ref)
        except SymrefLoop:
            raise BranchReferenceLoop(self)
        except NotGitRepository:
            raise NotBranchError(self.root_transport.base, controldir=self)
        if not nascent_ok and sha is None:
            raise NotBranchError(self.root_transport.base, controldir=self)
        return RemoteGitBranch(self, repo, ref_chain[-1], sha)

    def open_workingtree(self, recommend_upgrade=False):
        raise NotLocalUrl(self.transport.base)

    def has_workingtree(self):
        return False

    def get_peeled(self, name):
        return self.get_refs_container().get_peeled(name)

    def get_refs_container(self):
        if self._refs is not None:
            return self._refs
        result = self.fetch_pack(lambda x: None, None, lambda x: None, lambda x: trace.mutter('git: %s' % x))
        self._refs = remote_refs_dict_to_container(result.refs, result.symrefs)
        return self._refs

    def push_branch(self, source, revision_id=None, overwrite=False, remember=False, create_prefix=False, lossy=False, name=None, tag_selector=None):
        """Push the source branch into this ControlDir."""
        if revision_id is None:
            revision_id = source.last_revision()
        elif not source.repository.has_revision(revision_id):
            raise NoSuchRevision(source, revision_id)
        push_result = GitPushResult()
        push_result.workingtree_updated = None
        push_result.master_branch = None
        push_result.source_branch = source
        push_result.stacked_on = None
        push_result.branch_push_result = None
        repo = self.find_repository()
        refname = self._get_selected_ref(name)
        try:
            ref_chain, old_sha = self.get_refs_container().follow(refname)
        except NotBranchError:
            actual_refname = refname
            old_sha = None
        else:
            if ref_chain:
                actual_refname = ref_chain[-1]
            else:
                actual_refname = refname
        if isinstance(source, GitBranch) and lossy:
            raise errors.LossyPushToSameVCS(source.controldir, self)
        source_store = get_object_store(source.repository)
        fetch_tags = source.get_config_stack().get('branch.fetch_tags')

        def get_changed_refs(remote_refs):
            if self._refs is not None:
                update_refs_container(self._refs, remote_refs)
            ret = {}
            push_result.new_original_revid = revision_id
            if lossy:
                new_sha = source_store._lookup_revision_sha1(revision_id)
            else:
                try:
                    new_sha = repo.lookup_bzr_revision_id(revision_id)[0]
                except errors.NoSuchRevision:
                    raise errors.NoRoundtrippingSupport(source, self.open_branch(name=name, nascent_ok=True))
            old_sha = remote_refs.get(actual_refname)
            if not overwrite:
                if remote_divergence(old_sha, new_sha, source_store):
                    raise DivergedBranches(source, self.open_branch(name, nascent_ok=True))
            ret[actual_refname] = new_sha
            if fetch_tags:
                for tagname, revid in source.tags.get_tag_dict().items():
                    if tag_selector and (not tag_selector(tagname)):
                        continue
                    if lossy:
                        try:
                            new_sha = source_store._lookup_revision_sha1(revid)
                        except KeyError:
                            if source.repository.has_revision(revid):
                                raise
                    else:
                        try:
                            new_sha = repo.lookup_bzr_revision_id(revid)[0]
                        except errors.NoSuchRevision:
                            continue
                        else:
                            if not source.repository.has_revision(revid):
                                continue
                    ret[tag_name_to_ref(tagname)] = new_sha
            return ret
        with source_store.lock_read():

            def generate_pack_data(have, want, progress=None, ofs_delta=True):
                git_repo = getattr(source.repository, '_git', None)
                if git_repo:
                    shallow = git_repo.get_shallow()
                else:
                    shallow = None
                if lossy:
                    return source_store.generate_lossy_pack_data(have, want, shallow=shallow, progress=progress, ofs_delta=ofs_delta)
                elif shallow:
                    return source_store.generate_pack_data(have, want, shallow=shallow, progress=progress, ofs_delta=ofs_delta)
                else:
                    return source_store.generate_pack_data(have, want, progress=progress, ofs_delta=ofs_delta)
            dw_result = self.send_pack(get_changed_refs, generate_pack_data)
            new_refs = dw_result.refs
            error = dw_result.ref_status.get(actual_refname)
            if error:
                raise error
            for ref, error in dw_result.ref_status.items():
                if error:
                    trace.warning('unable to open ref %s: %s', ref, error)
        push_result.new_revid = repo.lookup_foreign_revision_id(new_refs[actual_refname])
        if old_sha is not None:
            push_result.old_revid = repo.lookup_foreign_revision_id(old_sha)
        else:
            push_result.old_revid = NULL_REVISION
        if self._refs is not None:
            update_refs_container(self._refs, new_refs)
        push_result.target_branch = self.open_branch(name)
        if old_sha is not None:
            push_result.branch_push_result = GitBranchPushResult()
            push_result.branch_push_result.source_branch = source
            push_result.branch_push_result.target_branch = push_result.target_branch
            push_result.branch_push_result.local_branch = None
            push_result.branch_push_result.master_branch = push_result.target_branch
            push_result.branch_push_result.old_revid = push_result.old_revid
            push_result.branch_push_result.new_revid = push_result.new_revid
            push_result.branch_push_result.new_original_revid = push_result.new_original_revid
        if source.get_push_location() is None or remember:
            source.set_push_location(push_result.target_branch.base)
        return push_result

    def _find_commondir(self):
        return self