import gc
from io import BytesIO
import logging
import os
import os.path as osp
import stat
import uuid
import git
from git.cmd import Git
from git.compat import defenc
from git.config import GitConfigParser, SectionConstraint, cp
from git.exc import (
from git.objects.base import IndexObject, Object
from git.objects.util import TraversableIterableObj
from git.util import (
from .util import (
from typing import Callable, Dict, Mapping, Sequence, TYPE_CHECKING, cast
from typing import Any, Iterator, Union
from git.types import Commit_ish, Literal, PathLike, TBD
class Submodule(IndexObject, TraversableIterableObj):
    """Implements access to a git submodule. They are special in that their sha
    represents a commit in the submodule's repository which is to be checked out
    at the path of this instance.

    The submodule type does not have a string type associated with it, as it exists
    solely as a marker in the tree and index.

    All methods work in bare and non-bare repositories.
    """
    _id_attribute_ = 'name'
    k_modules_file = '.gitmodules'
    k_head_option = 'branch'
    k_head_default = 'master'
    k_default_mode = stat.S_IFDIR | stat.S_IFLNK
    'Submodule flags. Submodules are directories with link-status.'
    type: Literal['submodule'] = 'submodule'
    'This is a bogus type for base class compatibility.'
    __slots__ = ('_parent_commit', '_url', '_branch_path', '_name', '__weakref__')
    _cache_attrs = ('path', '_url', '_branch_path')

    def __init__(self, repo: 'Repo', binsha: bytes, mode: Union[int, None]=None, path: Union[PathLike, None]=None, name: Union[str, None]=None, parent_commit: Union[Commit_ish, None]=None, url: Union[str, None]=None, branch_path: Union[PathLike, None]=None) -> None:
        """Initialize this instance with its attributes.

        We only document the parameters that differ from
        :class:`~git.objects.base.IndexObject`.

        :param repo: Our parent repository.
        :param binsha: Binary sha referring to a commit in the remote repository. See
            the ``url`` parameter.
        :param parent_commit: See :meth:`set_parent_commit`.
        :param url: The URL to the remote repository which is the submodule.
        :param branch_path: Full (relative) path to ref to checkout when cloning the
            remote repository.
        """
        super().__init__(repo, binsha, mode, path)
        self.size = 0
        self._parent_commit = parent_commit
        if url is not None:
            self._url = url
        if branch_path is not None:
            self._branch_path = branch_path
        if name is not None:
            self._name = name

    def _set_cache_(self, attr: str) -> None:
        if attr in ('path', '_url', '_branch_path'):
            reader: SectionConstraint = self.config_reader()
            try:
                self.path = reader.get('path')
            except cp.NoSectionError as e:
                if self.repo.working_tree_dir is not None:
                    raise ValueError("This submodule instance does not exist anymore in '%s' file" % osp.join(self.repo.working_tree_dir, '.gitmodules')) from e
            self._url = reader.get('url')
            self._branch_path = reader.get_value(self.k_head_option, git.Head.to_full_path(self.k_head_default))
        elif attr == '_name':
            raise AttributeError('Cannot retrieve the name of a submodule if it was not set initially')
        else:
            super()._set_cache_(attr)

    @classmethod
    def _get_intermediate_items(cls, item: 'Submodule') -> IterableList['Submodule']:
        """:return: all the submodules of our module repository"""
        try:
            return cls.list_items(item.module())
        except InvalidGitRepositoryError:
            return IterableList('')

    @classmethod
    def _need_gitfile_submodules(cls, git: Git) -> bool:
        return git.version_info[:3] >= (1, 7, 5)

    def __eq__(self, other: Any) -> bool:
        """Compare with another submodule."""
        return self._name == other._name

    def __ne__(self, other: object) -> bool:
        """Compare with another submodule for inequality."""
        return not self == other

    def __hash__(self) -> int:
        """Hash this instance using its logical id, not the sha."""
        return hash(self._name)

    def __str__(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return 'git.%s(name=%s, path=%s, url=%s, branch_path=%s)' % (type(self).__name__, self._name, self.path, self.url, self.branch_path)

    @classmethod
    def _config_parser(cls, repo: 'Repo', parent_commit: Union[Commit_ish, None], read_only: bool) -> SubmoduleConfigParser:
        """
        :return: Config Parser constrained to our submodule in read or write mode

        :raise IOError: If the .gitmodules file cannot be found, either locally or in
            the repository at the given parent commit. Otherwise the exception would be
            delayed until the first access of the config parser.
        """
        parent_matches_head = True
        if parent_commit is not None:
            try:
                parent_matches_head = repo.head.commit == parent_commit
            except ValueError:
                pass
        fp_module: Union[str, BytesIO]
        if not repo.bare and parent_matches_head and repo.working_tree_dir:
            fp_module = osp.join(repo.working_tree_dir, cls.k_modules_file)
        else:
            assert parent_commit is not None, 'need valid parent_commit in bare repositories'
            try:
                fp_module = cls._sio_modules(parent_commit)
            except KeyError as e:
                raise IOError('Could not find %s file in the tree of parent commit %s' % (cls.k_modules_file, parent_commit)) from e
        if not read_only and (repo.bare or not parent_matches_head):
            raise ValueError("Cannot write blobs of 'historical' submodule configurations")
        return SubmoduleConfigParser(fp_module, read_only=read_only)

    def _clear_cache(self) -> None:
        """Clear the possibly changed values."""
        for name in self._cache_attrs:
            try:
                delattr(self, name)
            except AttributeError:
                pass

    @classmethod
    def _sio_modules(cls, parent_commit: Commit_ish) -> BytesIO:
        """:return: Configuration file as BytesIO - we only access it through the respective blob's data"""
        sio = BytesIO(parent_commit.tree[cls.k_modules_file].data_stream.read())
        sio.name = cls.k_modules_file
        return sio

    def _config_parser_constrained(self, read_only: bool) -> SectionConstraint:
        """:return: Config Parser constrained to our submodule in read or write mode"""
        try:
            pc: Union['Commit_ish', None] = self.parent_commit
        except ValueError:
            pc = None
        parser = self._config_parser(self.repo, pc, read_only)
        parser.set_submodule(self)
        return SectionConstraint(parser, sm_section(self.name))

    @classmethod
    def _module_abspath(cls, parent_repo: 'Repo', path: PathLike, name: str) -> PathLike:
        if cls._need_gitfile_submodules(parent_repo.git):
            return osp.join(parent_repo.git_dir, 'modules', name)
        if parent_repo.working_tree_dir:
            return osp.join(parent_repo.working_tree_dir, path)
        raise NotADirectoryError()

    @classmethod
    def _clone_repo(cls, repo: 'Repo', url: str, path: PathLike, name: str, allow_unsafe_options: bool=False, allow_unsafe_protocols: bool=False, **kwargs: Any) -> 'Repo':
        """
        :return: Repo instance of newly cloned repository
        :param repo: Our parent repository
        :param url: URL to clone from
        :param path: Repository - relative path to the submodule checkout location
        :param name: Canonical name of the submodule
        :param allow_unsafe_protocols: Allow unsafe protocols to be used, like ext
        :param allow_unsafe_options: Allow unsafe options to be used, like --upload-pack
        :param kwargs: Additional arguments given to git.clone
        """
        module_abspath = cls._module_abspath(repo, path, name)
        module_checkout_path = module_abspath
        if cls._need_gitfile_submodules(repo.git):
            kwargs['separate_git_dir'] = module_abspath
            module_abspath_dir = osp.dirname(module_abspath)
            if not osp.isdir(module_abspath_dir):
                os.makedirs(module_abspath_dir)
            module_checkout_path = osp.join(str(repo.working_tree_dir), path)
        clone = git.Repo.clone_from(url, module_checkout_path, allow_unsafe_options=allow_unsafe_options, allow_unsafe_protocols=allow_unsafe_protocols, **kwargs)
        if cls._need_gitfile_submodules(repo.git):
            cls._write_git_file_and_module_config(module_checkout_path, module_abspath)
        return clone

    @classmethod
    def _to_relative_path(cls, parent_repo: 'Repo', path: PathLike) -> PathLike:
        """:return: a path guaranteed  to be relative to the given parent - repository
        :raise ValueError: if path is not contained in the parent repository's working tree"""
        path = to_native_path_linux(path)
        if path.endswith('/'):
            path = path[:-1]
        if osp.isabs(path) and parent_repo.working_tree_dir:
            working_tree_linux = to_native_path_linux(parent_repo.working_tree_dir)
            if not path.startswith(working_tree_linux):
                raise ValueError("Submodule checkout path '%s' needs to be within the parents repository at '%s'" % (working_tree_linux, path))
            path = path[len(working_tree_linux.rstrip('/')) + 1:]
            if not path:
                raise ValueError("Absolute submodule path '%s' didn't yield a valid relative path" % path)
        return path

    @classmethod
    def _write_git_file_and_module_config(cls, working_tree_dir: PathLike, module_abspath: PathLike) -> None:
        """Write a .git file containing a(preferably) relative path to the actual git module repository.

        It is an error if the module_abspath cannot be made into a relative path, relative to the working_tree_dir

        :note: This will overwrite existing files!
        :note: as we rewrite both the git file as well as the module configuration, we might fail on the configuration
            and will not roll back changes done to the git file. This should be a non - issue, but may easily be fixed
            if it becomes one.
        :param working_tree_dir: Directory to write the .git file into
        :param module_abspath: Absolute path to the bare repository
        """
        git_file = osp.join(working_tree_dir, '.git')
        rela_path = osp.relpath(module_abspath, start=working_tree_dir)
        if os.name == 'nt' and osp.isfile(git_file):
            os.remove(git_file)
        with open(git_file, 'wb') as fp:
            fp.write(('gitdir: %s' % rela_path).encode(defenc))
        with GitConfigParser(osp.join(module_abspath, 'config'), read_only=False, merge_includes=False) as writer:
            writer.set_value('core', 'worktree', to_native_path_linux(osp.relpath(working_tree_dir, start=module_abspath)))

    @classmethod
    def add(cls, repo: 'Repo', name: str, path: PathLike, url: Union[str, None]=None, branch: Union[str, None]=None, no_checkout: bool=False, depth: Union[int, None]=None, env: Union[Mapping[str, str], None]=None, clone_multi_options: Union[Sequence[TBD], None]=None, allow_unsafe_options: bool=False, allow_unsafe_protocols: bool=False) -> 'Submodule':
        """Add a new submodule to the given repository. This will alter the index
        as well as the .gitmodules file, but will not create a new commit.
        If the submodule already exists, no matter if the configuration differs
        from the one provided, the existing submodule will be returned.

        :param repo: Repository instance which should receive the submodule.
        :param name: The name/identifier for the submodule.
        :param path: Repository-relative or absolute path at which the submodule
            should be located.
            It will be created as required during the repository initialization.
        :param url: git-clone compatible URL, see git-clone reference for more information.
            If None, the repository is assumed to exist, and the url of the first
            remote is taken instead. This is useful if you want to make an existing
            repository a submodule of another one.
        :param branch: name of branch at which the submodule should (later) be checked out.
            The given branch must exist in the remote repository, and will be checked
            out locally as a tracking branch.
            It will only be written into the configuration if it not None, which is
            when the checked out branch will be the one the remote HEAD pointed to.
            The result you get in these situation is somewhat fuzzy, and it is recommended
            to specify at least 'master' here.
            Examples are 'master' or 'feature/new'.
        :param no_checkout: If True, and if the repository has to be cloned manually,
            no checkout will be performed.
        :param depth: Create a shallow clone with a history truncated to the
            specified number of commits.
        :param env: Optional dictionary containing the desired environment variables.
            Note: Provided variables will be used to update the execution environment
            for ``git``. If some variable is not specified in `env` and is defined in
            attr:`os.environ`, the value from attr:`os.environ` will be used. If you
            want to unset some variable, consider providing an empty string as its
            value.
        :param clone_multi_options: A list of Clone options. Please see
            :meth:`Repo.clone <git.repo.base.Repo.clone>` for details.
        :param allow_unsafe_protocols: Allow unsafe protocols to be used, like ext.
        :param allow_unsafe_options: Allow unsafe options to be used, like --upload-pack
        :return: The newly created submodule instance.
        :note: Works atomically, such that no change will be done if the repository
            update fails for instance.
        """
        if repo.bare:
            raise InvalidGitRepositoryError('Cannot add submodules to bare repositories')
        path = cls._to_relative_path(repo, path)
        if url is not None:
            url = to_native_path_linux(url)
        sm = cls(repo, cls.NULL_BIN_SHA, cls.k_default_mode, path, name, url='invalid-temporary')
        if sm.exists():
            try:
                sm = repo.head.commit.tree[str(path)]
                sm._name = name
                return sm
            except KeyError:
                index = repo.index
                entry = index.entries[index.entry_key(path, 0)]
                sm.binsha = entry.binsha
                return sm
        br = git.Head(repo, git.Head.to_full_path(str(branch) or cls.k_head_default))
        has_module = sm.module_exists()
        branch_is_default = branch is None
        if has_module and url is not None:
            if url not in [r.url for r in sm.module().remotes]:
                raise ValueError("Specified URL '%s' does not match any remote url of the repository at '%s'" % (url, sm.abspath))
        mrepo: Union[Repo, None] = None
        if url is None:
            if not has_module:
                raise ValueError('A URL was not given and a repository did not exist at %s' % path)
            mrepo = sm.module()
            urls = [r.url for r in mrepo.remotes]
            if not urls:
                raise ValueError("Didn't find any remote url in repository at %s" % sm.abspath)
            url = urls[0]
        else:
            kwargs: Dict[str, Union[bool, int, str, Sequence[TBD]]] = {'n': no_checkout}
            if not branch_is_default:
                kwargs['b'] = br.name
            if depth:
                if isinstance(depth, int):
                    kwargs['depth'] = depth
                else:
                    raise ValueError('depth should be an integer')
            if clone_multi_options:
                kwargs['multi_options'] = clone_multi_options
            mrepo = cls._clone_repo(repo, url, path, name, env=env, allow_unsafe_options=allow_unsafe_options, allow_unsafe_protocols=allow_unsafe_protocols, **kwargs)
        url = Git.polish_url(url)
        writer: Union[GitConfigParser, SectionConstraint]
        with sm.repo.config_writer() as writer:
            writer.set_value(sm_section(name), 'url', url)
        index = sm.repo.index
        with sm.config_writer(index=index, write=False) as writer:
            writer.set_value('url', url)
            writer.set_value('path', path)
            sm._url = url
            if not branch_is_default:
                writer.set_value(cls.k_head_option, br.path)
                sm._branch_path = br.path
        if mrepo:
            sm.binsha = mrepo.head.commit.binsha
        index.add([sm], write=True)
        return sm

    def update(self, recursive: bool=False, init: bool=True, to_latest_revision: bool=False, progress: Union['UpdateProgress', None]=None, dry_run: bool=False, force: bool=False, keep_going: bool=False, env: Union[Mapping[str, str], None]=None, clone_multi_options: Union[Sequence[TBD], None]=None, allow_unsafe_options: bool=False, allow_unsafe_protocols: bool=False) -> 'Submodule':
        """Update the repository of this submodule to point to the checkout
        we point at with the binsha of this instance.

        :param recursive:
            If True, we will operate recursively and update child modules as well.
        :param init:
            If True, the module repository will be cloned into place if necessary.
        :param to_latest_revision:
            If True, the submodule's sha will be ignored during checkout. Instead, the
            remote will be fetched, and the local tracking branch updated. This only
            works if we have a local tracking branch, which is the case if the remote
            repository had a master branch, or of the 'branch' option was specified for
            this submodule and the branch existed remotely.
        :param progress:
            UpdateProgress instance or None if no progress should be shown.
        :param dry_run:
            If True, the operation will only be simulated, but not performed.
            All performed operations are read-only.
        :param force:
            If True, we may reset heads even if the repository in question is dirty.
            Additionally we will be allowed to set a tracking branch which is ahead of
            its remote branch back into the past or the location of the remote branch.
            This will essentially 'forget' commits.
            If False, local tracking branches that are in the future of their respective
            remote branches will simply not be moved.
        :param keep_going:
            If True, we will ignore but log all errors, and keep going recursively.
            Unless dry_run is set as well, keep_going could cause subsequent / inherited
            errors you wouldn't see otherwise.
            In conjunction with dry_run, it can be useful to anticipate all errors when
            updating submodules.
        :param env: Optional dictionary containing the desired environment variables.
            Note: Provided variables will be used to update the execution environment
            for ``git``. If some variable is not specified in `env` and is defined in
            attr:`os.environ`, value from attr:`os.environ` will be used.
            If you want to unset some variable, consider providing the empty string as
            its value.
        :param clone_multi_options:
            List of Clone options.
            Please see :meth:`Repo.clone <git.repo.base.Repo.clone>` for details.
            They only take effect with the `init` option.
        :param allow_unsafe_protocols:
            Allow unsafe protocols to be used, like ext.
        :param allow_unsafe_options:
            Allow unsafe options to be used, like --upload-pack.

        :note: Does nothing in bare repositories.
        :note: This method is definitely not atomic if `recursive` is True.

        :return: self
        """
        if self.repo.bare:
            return self
        if progress is None:
            progress = UpdateProgress()
        prefix = ''
        if dry_run:
            prefix = 'DRY-RUN: '
        if dry_run:
            mrepo = None
        try:
            try:
                mrepo = self.module()
                rmts = mrepo.remotes
                len_rmts = len(rmts)
                for i, remote in enumerate(rmts):
                    op = FETCH
                    if i == 0:
                        op |= BEGIN
                    progress.update(op, i, len_rmts, prefix + 'Fetching remote %s of submodule %r' % (remote, self.name))
                    if not dry_run:
                        remote.fetch(progress=progress)
                    if i == len_rmts - 1:
                        op |= END
                    progress.update(op, i, len_rmts, prefix + 'Done fetching remote of submodule %r' % self.name)
            except InvalidGitRepositoryError:
                mrepo = None
                if not init:
                    return self
                checkout_module_abspath = self.abspath
                if not dry_run and osp.isdir(checkout_module_abspath):
                    try:
                        os.rmdir(checkout_module_abspath)
                    except OSError as e:
                        raise OSError('Module directory at %r does already exist and is non-empty' % checkout_module_abspath) from e
                progress.update(BEGIN | CLONE, 0, 1, prefix + "Cloning url '%s' to '%s' in submodule %r" % (self.url, checkout_module_abspath, self.name))
                if not dry_run:
                    mrepo = self._clone_repo(self.repo, self.url, self.path, self.name, n=True, env=env, multi_options=clone_multi_options, allow_unsafe_options=allow_unsafe_options, allow_unsafe_protocols=allow_unsafe_protocols)
                progress.update(END | CLONE, 0, 1, prefix + 'Done cloning to %s' % checkout_module_abspath)
                if not dry_run:
                    try:
                        mrepo = cast('Repo', mrepo)
                        remote_branch = find_first_remote_branch(mrepo.remotes, self.branch_name)
                        local_branch = mkhead(mrepo, self.branch_path)
                        local_branch.set_object(Object(mrepo, self.NULL_BIN_SHA))
                        mrepo.head.set_reference(local_branch, logmsg='submodule: attaching head to %s' % local_branch)
                        mrepo.head.reference.set_tracking_branch(remote_branch)
                    except (IndexError, InvalidGitRepositoryError):
                        _logger.warning('Failed to checkout tracking branch %s', self.branch_path)
                    with self.repo.config_writer() as writer:
                        writer.set_value(sm_section(self.name), 'url', self.url)
            binsha = self.binsha
            hexsha = self.hexsha
            if mrepo is not None:
                is_detached = mrepo.head.is_detached
            if mrepo is not None and to_latest_revision:
                msg_base = 'Cannot update to latest revision in repository at %r as ' % mrepo.working_dir
                if not is_detached:
                    rref = mrepo.head.reference.tracking_branch()
                    if rref is not None:
                        rcommit = rref.commit
                        binsha = rcommit.binsha
                        hexsha = rcommit.hexsha
                    else:
                        _logger.error("%s a tracking branch was not set for local branch '%s'", msg_base, mrepo.head.reference)
                else:
                    _logger.error('%s there was no local tracking branch', msg_base)
            if mrepo is not None and mrepo.head.commit.binsha != binsha:
                may_reset = True
                if mrepo.head.commit.binsha != self.NULL_BIN_SHA:
                    base_commit = mrepo.merge_base(mrepo.head.commit, hexsha)
                    if len(base_commit) == 0 or (base_commit[0] is not None and base_commit[0].hexsha == hexsha):
                        if force:
                            msg = 'Will force checkout or reset on local branch that is possibly in the future of'
                            msg += " the commit it will be checked out to, effectively 'forgetting' new commits"
                            _logger.debug(msg)
                        else:
                            msg = "Skipping %s on branch '%s' of submodule repo '%s' as it contains un-pushed commits"
                            msg %= (is_detached and 'checkout' or 'reset', mrepo.head, mrepo)
                            _logger.info(msg)
                            may_reset = False
                    if may_reset and (not force) and mrepo.is_dirty(index=True, working_tree=True, untracked_files=True):
                        raise RepositoryDirtyError(mrepo, 'Cannot reset a dirty repository')
                progress.update(BEGIN | UPDWKTREE, 0, 1, prefix + 'Updating working tree at %s for submodule %r to revision %s' % (self.path, self.name, hexsha))
                if not dry_run and may_reset:
                    if is_detached:
                        mrepo.git.checkout(hexsha, force=force)
                    else:
                        mrepo.head.reset(hexsha, index=True, working_tree=True)
                progress.update(END | UPDWKTREE, 0, 1, prefix + 'Done updating working tree for submodule %r' % self.name)
        except Exception as err:
            if not keep_going:
                raise
            _logger.error(str(err))
        if recursive:
            if mrepo is not None:
                for submodule in self.iter_items(self.module()):
                    submodule.update(recursive, init, to_latest_revision, progress=progress, dry_run=dry_run, force=force, keep_going=keep_going)
        return self

    @unbare_repo
    def move(self, module_path: PathLike, configuration: bool=True, module: bool=True) -> 'Submodule':
        """Move the submodule to a another module path. This involves physically moving
        the repository at our current path, changing the configuration, as well as
        adjusting our index entry accordingly.

        :param module_path: The path to which to move our module in the parent
            repository's working tree, given as repository - relative or absolute path.
            Intermediate directories will be created accordingly. If the path already
            exists, it must be empty. Trailing (back)slashes are removed automatically.
        :param configuration: If True, the configuration will be adjusted to let
            the submodule point to the given path.
        :param module: If True, the repository managed by this submodule
            will be moved as well. If False, we don't move the submodule's checkout,
            which may leave the parent repository in an inconsistent state.
        :return: self
        :raise ValueError: If the module path existed and was not empty, or was a file.
        :note: Currently the method is not atomic, and it could leave the repository
            in an inconsistent state if a sub-step fails for some reason.
        """
        if module + configuration < 1:
            raise ValueError('You must specify to move at least the module or the configuration of the submodule')
        module_checkout_path = self._to_relative_path(self.repo, module_path)
        if module_checkout_path == self.path:
            return self
        module_checkout_abspath = join_path_native(str(self.repo.working_tree_dir), module_checkout_path)
        if osp.isfile(module_checkout_abspath):
            raise ValueError('Cannot move repository onto a file: %s' % module_checkout_abspath)
        index = self.repo.index
        tekey = index.entry_key(module_checkout_path, 0)
        if configuration and tekey in index.entries:
            raise ValueError('Index entry for target path did already exist')
        if module:
            if osp.exists(module_checkout_abspath):
                if len(os.listdir(module_checkout_abspath)):
                    raise ValueError('Destination module directory was not empty')
                if osp.islink(module_checkout_abspath):
                    os.remove(module_checkout_abspath)
                else:
                    os.rmdir(module_checkout_abspath)
            else:
                pass
        cur_path = self.abspath
        renamed_module = False
        if module and osp.exists(cur_path):
            os.renames(cur_path, module_checkout_abspath)
            renamed_module = True
            if osp.isfile(osp.join(module_checkout_abspath, '.git')):
                module_abspath = self._module_abspath(self.repo, self.path, self.name)
                self._write_git_file_and_module_config(module_checkout_abspath, module_abspath)
        previous_sm_path = self.path
        try:
            if configuration:
                try:
                    ekey = index.entry_key(self.path, 0)
                    entry = index.entries[ekey]
                    del index.entries[ekey]
                    nentry = git.IndexEntry(entry[:3] + (module_checkout_path,) + entry[4:])
                    index.entries[tekey] = nentry
                except KeyError as e:
                    raise InvalidGitRepositoryError("Submodule's entry at %r did not exist" % self.path) from e
                with self.config_writer(index=index) as writer:
                    writer.set_value('path', module_checkout_path)
                    self.path = module_checkout_path
        except Exception:
            if renamed_module:
                os.renames(module_checkout_abspath, cur_path)
            raise
        if previous_sm_path == self.name:
            self.rename(module_checkout_path)
        return self

    @unbare_repo
    def remove(self, module: bool=True, force: bool=False, configuration: bool=True, dry_run: bool=False) -> 'Submodule':
        """Remove this submodule from the repository. This will remove our entry
        from the .gitmodules file and the entry in the .git/config file.

        :param module: If True, the checked out module we point to will be deleted as
            well. If that module is currently on a commit outside any branch in the
            remote, or if it is ahead of its tracking branch, or if there are modified
            or untracked files in its working tree, then the removal will fail. In case
            the removal of the repository fails for these reasons, the submodule status
            will not have been altered.
            If this submodule has child modules of its own, these will be deleted prior
            to touching the direct submodule.
        :param force: Enforces the deletion of the module even though it contains
            modifications. This basically enforces a brute-force file system based
            deletion.
        :param configuration: If True, the submodule is deleted from the configuration,
            otherwise it isn't. Although this should be enabled most of the time, this
            flag enables you to safely delete the repository of your submodule.
        :param dry_run: If True, we will not actually do anything, but throw the errors
            we would usually throw.
        :return: self
        :note: Doesn't work in bare repositories.
        :note: Doesn't work atomically, as failure to remove any part of the submodule
            will leave an inconsistent state.
        :raise InvalidGitRepositoryError: Thrown if the repository cannot be deleted.
        :raise OSError: If directories or files could not be removed.
        """
        if not (module or configuration):
            raise ValueError('Need to specify to delete at least the module, or the configuration')
        nc = 0
        for csm in self.children():
            nc += 1
            csm.remove(module, force, configuration, dry_run)
            del csm
        if configuration and (not dry_run) and (nc > 0):
            self.module().index.commit("Removed at least one of child-modules of '%s'" % self.name)
        if module and self.module_exists():
            mod = self.module()
            git_dir = mod.git_dir
            if force:
                mp = self.abspath
                method: Union[None, Callable[[PathLike], None]] = None
                if osp.islink(mp):
                    method = os.remove
                elif osp.isdir(mp):
                    method = rmtree
                elif osp.exists(mp):
                    raise AssertionError('Cannot forcibly delete repository as it was neither a link, nor a directory')
                if not dry_run:
                    assert method
                    method(mp)
            else:
                if mod.is_dirty(index=True, working_tree=True, untracked_files=True):
                    raise InvalidGitRepositoryError('Cannot delete module at %s with any modifications, unless force is specified' % mod.working_tree_dir)
                for remote in mod.remotes:
                    num_branches_with_new_commits = 0
                    rrefs = remote.refs
                    for rref in rrefs:
                        num_branches_with_new_commits += len(mod.git.cherry(rref)) != 0
                    if len(rrefs) and num_branches_with_new_commits == len(rrefs):
                        raise InvalidGitRepositoryError('Cannot delete module at %s as there are new commits' % mod.working_tree_dir)
                    if len(rrefs):
                        del rref
                    del rrefs
                    del remote
                if not dry_run:
                    self._clear_cache()
                    wtd = mod.working_tree_dir
                    del mod
                    gc.collect()
                    rmtree(str(wtd))
            if not dry_run and osp.isdir(git_dir):
                self._clear_cache()
                rmtree(git_dir)
        if not dry_run:
            self._clear_cache()
        if configuration and (not dry_run):
            parent_index = self.repo.index
            try:
                del parent_index.entries[parent_index.entry_key(self.path, 0)]
            except KeyError:
                pass
            parent_index.write()
            with self.repo.config_writer() as gcp_writer:
                gcp_writer.remove_section(sm_section(self.name))
            with self.config_writer() as sc_writer:
                sc_writer.remove_section()
        return self

    def set_parent_commit(self, commit: Union[Commit_ish, None], check: bool=True) -> 'Submodule':
        """Set this instance to use the given commit whose tree is supposed to
        contain the .gitmodules blob.

        :param commit:
            Commit-ish reference pointing at the root_tree, or None to always point to
            the most recent commit
        :param check:
            If True, relatively expensive checks will be performed to verify
            validity of the submodule.
        :raise ValueError: If the commit's tree didn't contain the .gitmodules blob.
        :raise ValueError:
            If the parent commit didn't store this submodule under the current path.
        :return: self
        """
        if commit is None:
            self._parent_commit = None
            return self
        pcommit = self.repo.commit(commit)
        pctree = pcommit.tree
        if self.k_modules_file not in pctree:
            raise ValueError('Tree of commit %s did not contain the %s file' % (commit, self.k_modules_file))
        prev_pc = self._parent_commit
        self._parent_commit = pcommit
        if check:
            parser = self._config_parser(self.repo, self._parent_commit, read_only=True)
            if not parser.has_section(sm_section(self.name)):
                self._parent_commit = prev_pc
                raise ValueError('Submodule at path %r did not exist in parent commit %s' % (self.path, commit))
        try:
            self.binsha = pctree[str(self.path)].binsha
        except KeyError:
            self.binsha = self.NULL_BIN_SHA
        self._clear_cache()
        return self

    @unbare_repo
    def config_writer(self, index: Union['IndexFile', None]=None, write: bool=True) -> SectionConstraint['SubmoduleConfigParser']:
        """
        :return: A config writer instance allowing you to read and write the data
            belonging to this submodule into the .gitmodules file.

        :param index: If not None, an IndexFile instance which should be written.
            Defaults to the index of the Submodule's parent repository.
        :param write: If True, the index will be written each time a configuration
            value changes.

        :note: The parameters allow for a more efficient writing of the index,
            as you can pass in a modified index on your own, prevent automatic writing,
            and write yourself once the whole operation is complete.

        :raise ValueError: If trying to get a writer on a parent_commit which does not
            match the current head commit.
        :raise IOError: If the .gitmodules file/blob could not be read
        """
        writer = self._config_parser_constrained(read_only=False)
        if index is not None:
            writer.config._index = index
        writer.config._auto_write = write
        return writer

    @unbare_repo
    def rename(self, new_name: str) -> 'Submodule':
        """Rename this submodule.

        :note:
            This method takes care of renaming the submodule in various places, such as:

            * $parent_git_dir / config
            * $working_tree_dir / .gitmodules
            * (git >= v1.8.0: move submodule repository to new name)

        As .gitmodules will be changed, you would need to make a commit afterwards. The
        changed .gitmodules file will already be added to the index.

        :return: This submodule instance
        """
        if self.name == new_name:
            return self
        with self.repo.config_writer() as pw:
            if pw.has_section(sm_section(self.name)):
                pw.rename_section(sm_section(self.name), sm_section(new_name))
        with self.config_writer(write=True).config as cw:
            cw.rename_section(sm_section(self.name), sm_section(new_name))
        self._name = new_name
        mod = self.module()
        if mod.has_separate_working_tree():
            destination_module_abspath = self._module_abspath(self.repo, self.path, new_name)
            source_dir = mod.git_dir
            if str(destination_module_abspath).startswith(str(mod.git_dir)):
                tmp_dir = self._module_abspath(self.repo, self.path, str(uuid.uuid4()))
                os.renames(source_dir, tmp_dir)
                source_dir = tmp_dir
            os.renames(source_dir, destination_module_abspath)
            if mod.working_tree_dir:
                self._write_git_file_and_module_config(mod.working_tree_dir, destination_module_abspath)
        return self

    @unbare_repo
    def module(self) -> 'Repo':
        """
        :return: Repo instance initialized from the repository at our submodule path

        :raise InvalidGitRepositoryError: If a repository was not available. This could
            also mean that it was not yet initialized.
        """
        module_checkout_abspath = self.abspath
        try:
            repo = git.Repo(module_checkout_abspath)
            if repo != self.repo:
                return repo
        except (InvalidGitRepositoryError, NoSuchPathError) as e:
            raise InvalidGitRepositoryError('No valid repository at %s' % module_checkout_abspath) from e
        else:
            raise InvalidGitRepositoryError('Repository at %r was not yet checked out' % module_checkout_abspath)

    def module_exists(self) -> bool:
        """:return: True if our module exists and is a valid git repository. See module() method."""
        try:
            self.module()
            return True
        except Exception:
            return False

    def exists(self) -> bool:
        """
        :return: True if the submodule exists, False otherwise. Please note that
            a submodule may exist (in the .gitmodules file) even though its module
            doesn't exist on disk.
        """
        loc = locals()
        for attr in self._cache_attrs:
            try:
                if hasattr(self, attr):
                    loc[attr] = getattr(self, attr)
            except (cp.NoSectionError, ValueError):
                pass
        self._clear_cache()
        try:
            try:
                self.path
                return True
            except Exception:
                return False
        finally:
            for attr in self._cache_attrs:
                if attr in loc:
                    setattr(self, attr, loc[attr])

    @property
    def branch(self) -> 'Head':
        """
        :return: The branch instance that we are to checkout

        :raise InvalidGitRepositoryError: If our module is not yet checked out
        """
        return mkhead(self.module(), self._branch_path)

    @property
    def branch_path(self) -> PathLike:
        """
        :return: Full (relative) path as string to the branch we would checkout
            from the remote and track
        """
        return self._branch_path

    @property
    def branch_name(self) -> str:
        """:return: The name of the branch, which is the shortest possible branch name"""
        return git.Head(self.repo, self._branch_path).name

    @property
    def url(self) -> str:
        """:return: The url to the repository which our module - repository refers to"""
        return self._url

    @property
    def parent_commit(self) -> 'Commit_ish':
        """
        :return: Commit instance with the tree containing the .gitmodules file

        :note: Will always point to the current head's commit if it was not set explicitly.
        """
        if self._parent_commit is None:
            return self.repo.commit()
        return self._parent_commit

    @property
    def name(self) -> str:
        """
        :return: The name of this submodule. It is used to identify it within the
            .gitmodules file.

        :note: By default, this is the name is the path at which to find the submodule,
            but in GitPython it should be a unique identifier similar to the identifiers
            used for remotes, which allows to change the path of the submodule easily.
        """
        return self._name

    def config_reader(self) -> SectionConstraint[SubmoduleConfigParser]:
        """
        :return: ConfigReader instance which allows you to query the configuration
            values of this submodule, as provided by the .gitmodules file.

        :note: The config reader will actually read the data directly from the
            repository and thus does not need nor care about your working tree.

        :note: Should be cached by the caller and only kept as long as needed.

        :raise IOError: If the .gitmodules file/blob could not be read.
        """
        return self._config_parser_constrained(read_only=True)

    def children(self) -> IterableList['Submodule']:
        """
        :return: IterableList(Submodule, ...) an iterable list of submodules instances
            which are children of this submodule or 0 if the submodule is not checked out.
        """
        return self._get_intermediate_items(self)

    @classmethod
    def iter_items(cls, repo: 'Repo', parent_commit: Union[Commit_ish, str]='HEAD', *Args: Any, **kwargs: Any) -> Iterator['Submodule']:
        """:return: Iterator yielding Submodule instances available in the given repository"""
        try:
            pc = repo.commit(parent_commit)
            parser = cls._config_parser(repo, pc, read_only=True)
        except (IOError, BadName):
            return
        for sms in parser.sections():
            n = sm_name(sms)
            p = parser.get(sms, 'path')
            u = parser.get(sms, 'url')
            b = cls.k_head_default
            if parser.has_option(sms, cls.k_head_option):
                b = str(parser.get(sms, cls.k_head_option))
            index = repo.index
            try:
                rt = pc.tree
                sm = rt[p]
            except KeyError:
                try:
                    entry = index.entries[index.entry_key(p, 0)]
                    sm = Submodule(repo, entry.binsha, entry.mode, entry.path)
                except KeyError:
                    continue
            if type(sm) is not git.objects.submodule.base.Submodule:
                continue
            sm._name = n
            if pc != repo.commit():
                sm._parent_commit = pc
            sm._branch_path = git.Head.to_full_path(b)
            sm._url = u
            yield sm