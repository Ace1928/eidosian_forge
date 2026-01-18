from io import BytesIO
from typing import TYPE_CHECKING, Optional, Union
from ..lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
from .. import errors, lockable_files
from .. import revision as _mod_revision
from .. import transport as _mod_transport
from .. import urlutils
from ..branch import (Branch, BranchFormat, BranchWriteLockResult,
from ..controldir import ControlDir
from ..decorators import only_raises
from ..lock import LogicalLockResult, _RelockDebugMixin
from ..trace import mutter
from . import bzrdir, rio
from .repository import MetaDirRepository
class BzrBranchFormat6(BranchFormatMetadir):
    """Branch format with last-revision and tags.

    Unlike previous formats, this has no explicit revision history. Instead,
    this just stores the last-revision, and the left-hand history leading
    up to there is the history.

    This format was introduced in bzr 0.15
    and became the default in 0.91.
    """

    def _branch_class(self):
        return BzrBranch6

    @classmethod
    def get_format_string(cls):
        """See BranchFormat.get_format_string()."""
        return b'Bazaar Branch Format 6 (bzr 0.15)\n'

    def get_format_description(self):
        """See BranchFormat.get_format_description()."""
        return 'Branch format 6'

    def initialize(self, a_controldir, name=None, repository=None, append_revisions_only=None):
        """Create a branch of this format in a_controldir."""
        utf8_files = [('last-revision', b'0 null:\n'), ('branch.conf', self._get_initial_config(append_revisions_only)), ('tags', b'')]
        return self._initialize_helper(a_controldir, utf8_files, name, repository)

    def make_tags(self, branch):
        """See breezy.branch.BranchFormat.make_tags()."""
        return _mod_tag.BasicTags(branch)

    def supports_set_append_revisions_only(self):
        return True
    supports_reference_locations = True