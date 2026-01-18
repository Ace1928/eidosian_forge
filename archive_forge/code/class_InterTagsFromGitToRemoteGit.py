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
class InterTagsFromGitToRemoteGit(InterTags):

    @classmethod
    def is_compatible(klass, source, target):
        if not isinstance(source, GitTags):
            return False
        if not isinstance(target, GitTags):
            return False
        if getattr(target.branch.repository, '_git', None) is not None:
            return False
        return True

    def merge(self, overwrite: bool=False, ignore_master: bool=False, selector: Optional[TagSelector]=None) -> Tuple[TagUpdates, Set[TagConflict]]:
        if self.source.branch.repository.has_same_location(self.target.branch.repository):
            return ({}, set())
        updates = {}
        conflicts = []
        source_tag_refs = self.source.branch.get_tag_refs()
        ref_to_tag_map = {}

        def get_changed_refs(old_refs):
            ret = dict(old_refs)
            for ref_name, tag_name, peeled, unpeeled in source_tag_refs.iteritems():
                if selector and (not selector(tag_name)):
                    continue
                if old_refs.get(ref_name) == unpeeled:
                    pass
                elif overwrite or ref_name not in old_refs:
                    ret[ref_name] = unpeeled
                    updates[tag_name] = self.target.branch.repository.lookup_foreign_revision_id(peeled)
                    ref_to_tag_map[ref_name] = tag_name
                    self.target.branch._tag_refs = None
                else:
                    conflicts.append((tag_name, self.source.branch.repository.lookup_foreign_revision_id(peeled), self.target.branch.repository.lookup_foreign_revision_id(old_refs[ref_name])))
            return ret
        result = self.target.branch.repository.controldir.send_pack(get_changed_refs, lambda have, want: [])
        if result is not None and (not isinstance(result, dict)):
            for ref, error in result.ref_status.items():
                if error:
                    warning('unable to update ref %s: %s', ref, error)
                    del updates[ref_to_tag_map[ref]]
        return (updates, set(conflicts))