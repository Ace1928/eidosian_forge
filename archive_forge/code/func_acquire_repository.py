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
def acquire_repository(self, make_working_trees=None, shared=False, possible_transports=None):
    """Implementation of RepositoryAcquisitionPolicy.acquire_repository

        Returns an existing repository to use.
        """
    return (self._repository, False)