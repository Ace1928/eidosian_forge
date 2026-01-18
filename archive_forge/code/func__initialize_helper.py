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
def _initialize_helper(self, a_controldir, utf8_files, name=None, repository=None):
    """Initialize a branch in a control dir, with specified files

        :param a_controldir: The bzrdir to initialize the branch in
        :param utf8_files: The files to create as a list of
            (filename, content) tuples
        :param name: Name of colocated branch to create, if any
        :return: a branch in this format
        """
    if name is None:
        name = a_controldir._get_selected_branch()
    mutter('creating branch %r in %s', self, a_controldir.user_url)
    branch_transport = a_controldir.get_branch_transport(self, name=name)
    control_files = lockable_files.LockableFiles(branch_transport, 'lock', lockdir.LockDir)
    control_files.create_lock()
    control_files.lock_write()
    try:
        utf8_files += [('format', self.as_string())]
        for filename, content in utf8_files:
            branch_transport.put_bytes(filename, content, mode=a_controldir._get_file_mode())
    finally:
        control_files.unlock()
    branch = self.open(a_controldir, name, _found=True, found_repository=repository)
    self._run_post_branch_init_hooks(a_controldir, name, branch)
    return branch