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
class InterGitLocalGitBranch(InterGitBranch):
    """InterBranch that copies from a remote to a local git branch."""

    @staticmethod
    def _get_branch_formats_to_test():
        from .remote import RemoteGitBranchFormat
        return [(RemoteGitBranchFormat(), LocalGitBranchFormat()), (LocalGitBranchFormat(), LocalGitBranchFormat())]

    @classmethod
    def is_compatible(self, source, target):
        return isinstance(source, GitBranch) and isinstance(target, LocalGitBranch)

    def fetch(self, stop_revision=None, fetch_tags=None, limit=None, lossy=False):
        if lossy:
            raise errors.LossyPushToSameVCS(source_branch=self.source, target_branch=self.target)
        interrepo = _mod_repository.InterRepository.get(self.source.repository, self.target.repository)
        if stop_revision is None:
            stop_revision = self.source.last_revision()
        if fetch_tags is None:
            c = self.source.get_config_stack()
            fetch_tags = c.get('branch.fetch_tags')
        determine_wants = interrepo.get_determine_wants_revids([stop_revision], include_tags=fetch_tags)
        interrepo.fetch_objects(determine_wants, limit=limit)
        return _mod_repository.FetchResult()

    def _basic_push(self, overwrite=False, stop_revision=None, tag_selector=None):
        if overwrite is True:
            overwrite = {'history', 'tags'}
        elif not overwrite:
            overwrite = set()
        result = GitBranchPushResult()
        result.source_branch = self.source
        result.target_branch = self.target
        result.old_revid = self.target.last_revision()
        refs, stop_revision = self.update_refs(stop_revision)
        _update_tip(self.source, self.target, stop_revision, 'history' in overwrite)
        tags_ret = self.source.tags.merge_to(self.target.tags, overwrite='tags' in overwrite, selector=tag_selector)
        if isinstance(tags_ret, tuple):
            result.tag_updates, result.tag_conflicts = tags_ret
        else:
            result.tag_conflicts = tags_ret
        result.new_revid = self.target.last_revision()
        return result

    def update_refs(self, stop_revision=None):
        interrepo = _mod_repository.InterRepository.get(self.source.repository, self.target.repository)
        c = self.source.get_config_stack()
        fetch_tags = c.get('branch.fetch_tags')
        if stop_revision is None:
            result = interrepo.fetch(branches=[self.source.ref], include_tags=fetch_tags)
            try:
                head = result.refs[self.source.ref]
            except KeyError:
                stop_revision = revision.NULL_REVISION
            else:
                stop_revision = self.target.lookup_foreign_revision_id(head)
        else:
            result = interrepo.fetch(revision_id=stop_revision, include_tags=fetch_tags)
        return (result.refs, stop_revision)

    def pull(self, stop_revision=None, overwrite=False, possible_transports=None, run_hooks=True, local=False, tag_selector=None):
        if local:
            raise errors.LocalRequiresBoundBranch()
        if overwrite is True:
            overwrite = {'history', 'tags'}
        elif not overwrite:
            overwrite = set()
        result = GitPullResult()
        result.source_branch = self.source
        result.target_branch = self.target
        with self.target.lock_write(), self.source.lock_read():
            result.old_revid = self.target.last_revision()
            refs, stop_revision = self.update_refs(stop_revision)
            _update_tip(self.source, self.target, stop_revision, 'history' in overwrite)
            tags_ret = self.source.tags.merge_to(self.target.tags, overwrite='tags' in overwrite, selector=tag_selector)
            if isinstance(tags_ret, tuple):
                result.tag_updates, result.tag_conflicts = tags_ret
            else:
                result.tag_conflicts = tags_ret
            result.new_revid = self.target.last_revision()
            result.local_branch = None
            result.master_branch = result.target_branch
            if run_hooks:
                for hook in branch.Branch.hooks['post_pull']:
                    hook(result)
        return result