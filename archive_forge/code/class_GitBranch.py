import contextlib
from collections import defaultdict
from functools import partial
from io import BytesIO
from typing import Dict, Optional, Set, Tuple
from dulwich.config import ConfigFile as GitConfigFile
from dulwich.config import parse_submodules
from dulwich.objects import ZERO_SHA, NotCommitError
from dulwich.repo import check_ref_format
from .. import branch, config, controldir, errors, lock
from .. import repository as _mod_repository
from .. import revision, trace, transport, urlutils
from ..foreign import ForeignBranch
from ..revision import NULL_REVISION
from ..tag import InterTags, TagConflict, Tags, TagSelector, TagUpdates
from ..trace import is_quiet, mutter, warning
from .errors import NoPushSupport
from .mapping import decode_git_path, encode_git_path
from .push import remote_divergence
from .refs import (branch_name_to_ref, is_tag, ref_to_branch_name,
from .unpeel_map import UnpeelMap
from .urls import bzr_url_to_git_url, git_url_to_bzr_url
class GitBranch(ForeignBranch):
    """An adapter to git repositories for bzr Branch objects."""

    @property
    def control_transport(self):
        return self._control_transport

    @property
    def user_transport(self):
        return self._user_transport

    def __init__(self, controldir, repository, ref: bytes, format):
        self.repository = repository
        self._format = format
        self.controldir = controldir
        self._lock_mode = None
        self._lock_count = 0
        super().__init__(repository.get_mapping())
        if not isinstance(ref, bytes):
            raise TypeError('ref is invalid: %r' % ref)
        self.ref = ref
        self._head = None
        self._user_transport = controldir.user_transport.clone('.')
        self._control_transport = controldir.control_transport.clone('.')
        self._tag_refs = None
        params: Dict[str, str] = {}
        try:
            self.name = ref_to_branch_name(ref)
        except ValueError:
            self.name = None
            if self.ref is not None:
                params = {'ref': urlutils.escape(self.ref, safe='')}
        else:
            if self.name:
                params = {'branch': urlutils.escape(self.name, safe='')}
        for k, v in params.items():
            self._user_transport.set_segment_parameter(k, v)
            self._control_transport.set_segment_parameter(k, v)
        self.base = controldir.user_transport.base

    def _get_checkout_format(self, lightweight=False):
        """Return the most suitable metadir for a checkout of this branch.
        Weaves are used if this branch's repository uses weaves.
        """
        if lightweight:
            return controldir.format_registry.make_controldir('git')
        else:
            return controldir.format_registry.make_controldir('default')

    def set_stacked_on_url(self, url):
        raise branch.UnstackableBranchFormat(self._format, self.base)

    def get_child_submit_format(self):
        """Return the preferred format of submissions to this branch."""
        ret = self.get_config_stack().get('child_submit_format')
        if ret is not None:
            return ret
        return 'git'

    def get_config(self):
        from .config import GitBranchConfig
        return GitBranchConfig(self)

    def get_config_stack(self):
        from .config import GitBranchStack
        return GitBranchStack(self)

    def _get_nick(self, local=False, possible_master_transports=None):
        """Find the nick name for this branch.

        :return: Branch nick
        """
        if getattr(self.repository, '_git', None):
            cs = self.repository._git.get_config_stack()
            try:
                return cs.get((b'branch', self.name.encode('utf-8')), b'nick').decode('utf-8')
            except KeyError:
                pass
        return self.name or 'HEAD'

    def _set_nick(self, nick):
        cf = self.repository._git.get_config()
        cf.set((b'branch', self.name.encode('utf-8')), b'nick', nick.encode('utf-8'))
        f = BytesIO()
        cf.write_to_file(f)
        self.repository._git._put_named_file('config', f.getvalue())
    nick = property(_get_nick, _set_nick)

    def __repr__(self):
        return '<{}({!r}, {!r})>'.format(self.__class__.__name__, self.repository.base, self.name)

    def set_last_revision(self, revid):
        raise NotImplementedError(self.set_last_revision)

    def generate_revision_history(self, revid, last_rev=None, other_branch=None):
        if last_rev is not None:
            graph = self.repository.get_graph()
            if not graph.is_ancestor(last_rev, revid):
                raise errors.DivergedBranches(self, other_branch)
        self.set_last_revision(revid)

    def lock_write(self, token=None):
        if token is not None:
            raise errors.TokenLockingNotSupported(self)
        if self._lock_mode:
            if self._lock_mode == 'r':
                raise errors.ReadOnlyError(self)
            self._lock_count += 1
        else:
            self._lock_ref()
            self._lock_mode = 'w'
            self._lock_count = 1
        self.repository.lock_write()
        return lock.LogicalLockResult(self.unlock)

    def leave_lock_in_place(self):
        raise NotImplementedError(self.leave_lock_in_place)

    def dont_leave_lock_in_place(self):
        raise NotImplementedError(self.dont_leave_lock_in_place)

    def get_stacked_on_url(self):
        raise branch.UnstackableBranchFormat(self._format, self.base)

    def _get_push_origin(self, cs):
        """Get the name for the push origin.

        The exact behaviour is documented in the git-config(1) manpage.
        """
        try:
            return cs.get((b'branch', self.name.encode('utf-8')), b'pushRemote')
        except KeyError:
            try:
                return cs.get((b'branch',), b'remote')
            except KeyError:
                try:
                    return cs.get((b'branch', self.name.encode('utf-8')), b'remote')
                except KeyError:
                    return b'origin'

    def _get_origin(self, cs):
        try:
            return cs.get((b'branch', self.name.encode('utf-8')), b'remote')
        except KeyError:
            return b'origin'

    def _get_related_push_branch(self, cs):
        remote = self._get_push_origin(cs)
        try:
            location = cs.get((b'remote', remote), b'url')
        except KeyError:
            return None
        return git_url_to_bzr_url(location.decode('utf-8'), ref=self.ref)

    def _get_related_merge_branch(self, cs):
        remote = self._get_origin(cs)
        try:
            location = cs.get((b'remote', remote), b'url')
        except KeyError:
            return None
        try:
            ref = cs.get((b'branch', remote), b'merge')
        except KeyError:
            ref = b'HEAD'
        return git_url_to_bzr_url(location.decode('utf-8'), ref=ref)

    def _get_parent_location(self):
        """See Branch.get_parent()."""
        cs = self.repository._git.get_config_stack()
        return self._get_related_merge_branch(cs)

    def set_parent(self, location):
        cs = self.repository._git.get_config()
        remote = self._get_origin(cs)
        this_url = urlutils.strip_segment_parameters(self.user_url)
        target_url, branch, ref = bzr_url_to_git_url(location)
        location = urlutils.relative_url(this_url, target_url)
        cs.set((b'remote', remote), b'url', location)
        cs.set((b'remote', remote), b'fetch', b'+refs/heads/*:refs/remotes/%s/*' % remote)
        if self.name:
            if branch:
                cs.set((b'branch', self.name.encode()), b'merge', branch_name_to_ref(branch))
            elif ref:
                cs.set((b'branch', self.name.encode()), b'merge', ref)
            else:
                cs.set((b'branch', self.name.encode()), b'merge', b'HEAD')
        self.repository._write_git_config(cs)

    def break_lock(self):
        raise NotImplementedError(self.break_lock)

    def lock_read(self):
        if self._lock_mode:
            if self._lock_mode not in ('r', 'w'):
                raise ValueError(self._lock_mode)
            self._lock_count += 1
        else:
            self._lock_mode = 'r'
            self._lock_count = 1
        self.repository.lock_read()
        return lock.LogicalLockResult(self.unlock)

    def peek_lock_mode(self):
        return self._lock_mode

    def is_locked(self):
        return self._lock_mode is not None

    def _lock_ref(self):
        pass

    def _unlock_ref(self):
        pass

    def unlock(self):
        """See Branch.unlock()."""
        if self._lock_count == 0:
            raise errors.LockNotHeld(self)
        try:
            self._lock_count -= 1
            if self._lock_count == 0:
                if self._lock_mode == 'w':
                    self._unlock_ref()
                self._lock_mode = None
                self._clear_cached_state()
        finally:
            self.repository.unlock()

    def get_physical_lock_status(self):
        return False

    def last_revision(self):
        with self.lock_read():
            if self.head is None:
                return revision.NULL_REVISION
            return self.lookup_foreign_revision_id(self.head)

    def _basic_push(self, target, overwrite=False, stop_revision=None, tag_selector=None):
        return branch.InterBranch.get(self, target)._basic_push(overwrite, stop_revision, tag_selector=tag_selector)

    def lookup_foreign_revision_id(self, foreign_revid):
        try:
            return self.repository.lookup_foreign_revision_id(foreign_revid, self.mapping)
        except KeyError:
            return self.mapping.revision_id_foreign_to_bzr(foreign_revid)

    def lookup_bzr_revision_id(self, revid):
        return self.repository.lookup_bzr_revision_id(revid, mapping=self.mapping)

    def get_unshelver(self, tree):
        raise errors.StoringUncommittedNotSupported(self)

    def _clear_cached_state(self):
        super()._clear_cached_state()
        self._tag_refs = None

    def _iter_tag_refs(self, refs):
        """Iterate over the tag refs.

        :param refs: Refs dictionary (name -> git sha1)
        :return: iterator over (ref_name, tag_name, peeled_sha1, unpeeled_sha1)
        """
        raise NotImplementedError(self._iter_tag_refs)

    def get_tag_refs(self):
        with self.lock_read():
            if self._tag_refs is None:
                self._tag_refs = list(self._iter_tag_refs())
            return self._tag_refs

    def import_last_revision_info_and_tags(self, source, revno, revid, lossy=False):
        """Set the last revision info, importing from another repo if necessary.

        This is used by the bound branch code to upload a revision to
        the master branch first before updating the tip of the local branch.
        Revisions referenced by source's tags are also transferred.

        :param source: Source branch to optionally fetch from
        :param revno: Revision number of the new tip
        :param revid: Revision id of the new tip
        :param lossy: Whether to discard metadata that can not be
            natively represented
        :return: Tuple with the new revision number and revision id
            (should only be different from the arguments when lossy=True)
        """
        push_result = source.push(self, stop_revision=revid, lossy=lossy, _stop_revno=revno)
        return (push_result.new_revno, push_result.new_revid)

    def reconcile(self, thorough=True):
        """Make sure the data stored in this branch is consistent."""
        from ..reconcile import ReconcileResult
        return ReconcileResult()