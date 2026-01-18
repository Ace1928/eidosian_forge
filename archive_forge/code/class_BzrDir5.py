from io import BytesIO
from ... import errors, lockable_files
from ...bzr.bzrdir import BzrDir, BzrDirFormat, BzrDirMetaFormat1
from ...controldir import (ControlDir, Converter, MustHaveWorkingTree,
from ...i18n import gettext
from ...lazy_import import lazy_import
from ...transport import NoSuchFile, get_transport, local
import os
from breezy import (
from breezy.bzr import (
from breezy.plugins.weave_fmt.store.versioned import VersionedFileStore
from breezy.transactions import WriteTransaction
from breezy.plugins.weave_fmt import xml4
class BzrDir5(BzrDirPreSplitOut):
    """A .bzr version 5 control object.

    This is a deprecated format and may be removed after sept 2006.
    """

    def has_workingtree(self):
        """See ControlDir.has_workingtree."""
        return True

    def open_repository(self):
        """See ControlDir.open_repository."""
        from .repository import RepositoryFormat5
        return RepositoryFormat5().open(self, _found=True)

    def open_workingtree(self, unsupported=False, recommend_upgrade=True):
        """See ControlDir.create_workingtree."""
        from .workingtree import WorkingTreeFormat2
        wt_format = WorkingTreeFormat2()
        return wt_format.open(self, _found=True)