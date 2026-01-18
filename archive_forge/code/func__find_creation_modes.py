import contextlib
import os
from dulwich.refs import SymrefLoop
from .. import branch as _mod_branch
from .. import errors as brz_errors
from .. import osutils, trace, urlutils
from ..controldir import (BranchReferenceLoop, ControlDir, ControlDirFormat,
from ..transport import (FileExists, NoSuchFile, do_catching_redirections,
from .mapping import decode_git_path, encode_git_path
from .push import GitPushResult
from .transportgit import OBJECTDIR, TransportObjectStore
def _find_creation_modes(self):
    """Determine the appropriate modes for files and directories.

        They're always set to be consistent with the base directory,
        assuming that this transport allows setting modes.
        """
    if self._mode_check_done:
        return
    self._mode_check_done = True
    try:
        st = self.transport.stat('.')
    except brz_errors.TransportNotPossible:
        self._dir_mode = None
        self._file_mode = None
    else:
        if st.st_mode & 4095 == 0:
            self._dir_mode = None
            self._file_mode = None
        else:
            self._dir_mode = st.st_mode & 4095 | 448
            self._file_mode = self._dir_mode & ~3657