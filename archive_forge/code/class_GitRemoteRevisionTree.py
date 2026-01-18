import gzip
import re
from dulwich.refs import SymrefLoop
from .. import config, debug, errors, osutils, trace, ui, urlutils
from ..controldir import BranchReferenceLoop
from ..errors import (AlreadyBranchError, BzrError, ConnectionReset,
from ..push import PushResult
from ..revision import NULL_REVISION
from ..revisiontree import RevisionTree
from ..transport import (NoSuchFile, Transport,
from . import is_github_url, lazy_check_versions, user_agent_for_github
import os
import select
import urllib.parse as urlparse
import dulwich
import dulwich.client
from dulwich.errors import GitProtocolError, HangupException
from dulwich.pack import (PACK_SPOOL_FILE_MAX_SIZE, Pack, load_pack_index,
from dulwich.protocol import ZERO_SHA
from dulwich.refs import SYMREF, DictRefsContainer
from dulwich.repo import NotGitRepository
from .branch import (GitBranch, GitBranchFormat, GitBranchPushResult, GitTags,
from .dir import GitControlDirFormat, GitDir
from .errors import GitSmartRemoteNotSupported
from .mapping import encode_git_path, mapping_registry
from .object_store import get_object_store
from .push import remote_divergence
from .refs import (branch_name_to_ref, is_peeled, ref_to_tag_name,
from .repository import GitRepository, GitRepositoryFormat
class GitRemoteRevisionTree(RevisionTree):

    def archive(self, format, name, root=None, subdir=None, force_mtime=None, recurse_nested=False):
        """Create an archive of this tree.

        :param format: Format name (e.g. 'tar')
        :param name: target file name
        :param root: Root directory name (or None)
        :param subdir: Subdirectory to export (or None)
        :return: Iterator over archive chunks
        """
        if recurse_nested:
            raise NotImplementedError('recurse_nested is not yet supported')
        commit = self._repository.lookup_bzr_revision_id(self.get_revision_id())[0]
        from tempfile import SpooledTemporaryFile
        f = SpooledTemporaryFile(max_size=PACK_SPOOL_FILE_MAX_SIZE, prefix='incoming-')
        reverse_refs = {v: k for k, v in self._repository.controldir.get_refs_container().as_dict().items()}
        try:
            committish = reverse_refs[commit]
        except KeyError:
            committish = commit
        self._repository.archive(format, committish, f.write, subdirs=[subdir] if subdir else None, prefix=root + '/' if root else '')
        f.seek(0)
        return osutils.file_iterator(f)

    def is_versioned(self, path):
        raise GitSmartRemoteNotSupported(self.is_versioned, self)

    def has_filename(self, path):
        raise GitSmartRemoteNotSupported(self.has_filename, self)

    def get_file_text(self, path):
        raise GitSmartRemoteNotSupported(self.get_file_text, self)

    def list_files(self, include_root=False, from_dir=None, recursive=True):
        raise GitSmartRemoteNotSupported(self.list_files, self)