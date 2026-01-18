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
def _basic_pull(self, stop_revision, overwrite, run_hooks, _override_hook_target, _hook_master, tag_selector=None):
    if overwrite is True:
        overwrite = {'history', 'tags'}
    elif not overwrite:
        overwrite = set()
    result = GitBranchPullResult()
    result.source_branch = self.source
    if _override_hook_target is None:
        result.target_branch = self.target
    else:
        result.target_branch = _override_hook_target
    with self.target.lock_write(), self.source.lock_read():
        result.old_revno, result.old_revid = self.target.last_revision_info()
        result.new_git_head, remote_refs = self._update_revisions(stop_revision, overwrite='history' in overwrite, tag_selector=tag_selector)
        tags_ret = self.source.tags.merge_to(self.target.tags, 'tags' in overwrite, ignore_master=True)
        if isinstance(tags_ret, tuple):
            result.tag_updates, result.tag_conflicts = tags_ret
        else:
            result.tag_conflicts = tags_ret
        result.new_revno, result.new_revid = self.target.last_revision_info()
        self.update_references(revid=result.new_revid)
        if _hook_master:
            result.master_branch = _hook_master
            result.local_branch = result.target_branch
        else:
            result.master_branch = result.target_branch
            result.local_branch = None
        if run_hooks:
            for hook in branch.Branch.hooks['post_pull']:
                hook(result)
        return result