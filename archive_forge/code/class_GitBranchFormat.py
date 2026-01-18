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
class GitBranchFormat(branch.BranchFormat):

    def network_name(self):
        return b'git'

    def supports_tags(self):
        return True

    def supports_leaving_lock(self):
        return False

    def supports_tags_referencing_ghosts(self):
        return False

    def tags_are_versioned(self):
        return False

    def get_foreign_tests_branch_factory(self):
        from .tests.test_branch import ForeignTestsBranchFactory
        return ForeignTestsBranchFactory()

    def make_tags(self, branch):
        try:
            return branch.tags
        except AttributeError:
            pass
        if getattr(branch.repository, '_git', None) is None:
            from .remote import RemoteGitTagDict
            return RemoteGitTagDict(branch)
        else:
            return LocalGitTagDict(branch)

    def initialize(self, a_controldir, name=None, repository=None, append_revisions_only=None):
        raise NotImplementedError(self.initialize)

    def get_reference(self, controldir, name=None):
        return controldir.get_branch_reference(name=name)

    def set_reference(self, controldir, name, target):
        return controldir.set_branch_reference(target, name)

    def stores_revno(self):
        """True if this branch format store revision numbers."""
        return False
    supports_reference_locations = False