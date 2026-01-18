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
class InterLocalGitRemoteGitBranch(InterGitBranch):
    """InterBranch that copies from a local to a remote git branch."""

    @staticmethod
    def _get_branch_formats_to_test():
        from .remote import RemoteGitBranchFormat
        return [(LocalGitBranchFormat(), RemoteGitBranchFormat())]

    @classmethod
    def is_compatible(self, source, target):
        from .remote import RemoteGitBranch
        return isinstance(source, LocalGitBranch) and isinstance(target, RemoteGitBranch)

    def _basic_push(self, overwrite, stop_revision, tag_selector=None):
        from .remote import parse_git_error
        result = GitBranchPushResult()
        result.source_branch = self.source
        result.target_branch = self.target
        if stop_revision is None:
            stop_revision = self.source.last_revision()

        def get_changed_refs(old_refs):
            old_ref = old_refs.get(self.target.ref, None)
            if old_ref is None:
                result.old_revid = revision.NULL_REVISION
            else:
                result.old_revid = self.target.lookup_foreign_revision_id(old_ref)
            new_ref = self.source.repository.lookup_bzr_revision_id(stop_revision)[0]
            if not overwrite:
                if remote_divergence(old_ref, new_ref, self.source.repository._git.object_store):
                    raise errors.DivergedBranches(self.source, self.target)
            refs = {self.target.ref: new_ref}
            result.new_revid = stop_revision
            for name, sha in self.source.repository._git.refs.as_dict(b'refs/tags').items():
                if tag_selector and (not tag_selector(name.decode('utf-8'))):
                    continue
                if sha not in self.source.repository._git:
                    trace.mutter('Ignoring missing SHA: %s', sha)
                    continue
                refs[tag_name_to_ref(name)] = sha
            return refs
        dw_result = self.target.repository.send_pack(get_changed_refs, self.source.repository._git.generate_pack_data)
        if dw_result is not None and (not isinstance(dw_result, dict)):
            error = dw_result.ref_status.get(self.target.ref)
            if error:
                raise parse_git_error(self.target.user_url, error)
            for ref, error in dw_result.ref_status.items():
                if error:
                    trace.warning('unable to open ref %s: %s', ref, error)
        return result