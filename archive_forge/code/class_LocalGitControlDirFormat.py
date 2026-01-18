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
class LocalGitControlDirFormat(GitControlDirFormat):
    """The .git directory control format."""
    bare = False

    @classmethod
    def _known_formats(self):
        return {LocalGitControlDirFormat()}

    @property
    def repository_format(self):
        from .repository import GitRepositoryFormat
        return GitRepositoryFormat()

    @property
    def workingtree_format(self):
        from .workingtree import GitWorkingTreeFormat
        return GitWorkingTreeFormat()

    def get_branch_format(self):
        from .branch import LocalGitBranchFormat
        return LocalGitBranchFormat()

    def open(self, transport, _found=None):
        """Open this directory.

        """
        from .transportgit import TransportRepo

        def _open(transport):
            try:
                return TransportRepo(transport, self.bare, refs_text=getattr(self, '_refs_text', None))
            except ValueError as e:
                if e.args == ("Expected file to start with 'gitdir: '",):
                    raise brz_errors.NotBranchError(path=transport.base)
                raise

        def redirected(transport, e, redirection_notice):
            trace.note(redirection_notice)
            return transport._redirected_to(e.source, e.target)
        gitrepo = do_catching_redirections(_open, transport, redirected)
        if not _found and (not gitrepo._controltransport.has('objects')):
            raise brz_errors.NotBranchError(path=transport.base)
        return LocalGitDir(transport, gitrepo, self)

    def get_format_description(self):
        return 'Local Git Repository'

    def initialize_on_transport(self, transport):
        from .transportgit import TransportRepo
        git_repo = TransportRepo.init(transport, bare=self.bare)
        return LocalGitDir(transport, git_repo, self)

    def initialize_on_transport_ex(self, transport, use_existing_dir=False, create_prefix=False, force_new_repo=False, stacked_on=None, stack_on_pwd=None, repo_format_name=None, make_working_trees=None, shared_repo=False, vfs_only=False):
        if shared_repo:
            raise brz_errors.SharedRepositoriesUnsupported(self)

        def make_directory(transport):
            transport.mkdir('.')
            return transport

        def redirected(transport, e, redirection_notice):
            trace.note(redirection_notice)
            return transport._redirected_to(e.source, e.target)
        try:
            transport = do_catching_redirections(make_directory, transport, redirected)
        except FileExists:
            if not use_existing_dir:
                raise
        except NoSuchFile:
            if not create_prefix:
                raise
            transport.create_prefix()
        controldir = self.initialize_on_transport(transport)
        if repo_format_name:
            result_repo = controldir.find_repository()
            repository_policy = UseExistingRepository(result_repo)
            result_repo.lock_write()
        else:
            result_repo = None
            repository_policy = None
        return (result_repo, controldir, False, repository_policy)

    def is_supported(self):
        return True

    def supports_transport(self, transport):
        try:
            external_url = transport.external_url()
        except brz_errors.InProcessTransport:
            raise brz_errors.NotBranchError(path=transport.base)
        return external_url.startswith('file:')

    def is_control_filename(self, filename):
        return filename == '.git' or filename.startswith('.git/') or filename.startswith('.git\\')