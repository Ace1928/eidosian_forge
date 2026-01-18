import os
from git.compat import defenc
from git.objects import Object
from git.objects.commit import Commit
from git.util import (
from gitdb.exc import BadObject, BadName
from .log import RefLog
from typing import (
from git.types import Commit_ish, PathLike
class SymbolicReference:
    """Special case of a reference that is symbolic.

    This does not point to a specific commit, but to another
    :class:`~git.refs.head.Head`, which itself specifies a commit.

    A typical example for a symbolic reference is ``HEAD``.
    """
    __slots__ = ('repo', 'path')
    _resolve_ref_on_create = False
    _points_to_commits_only = True
    _common_path_default = ''
    _remote_common_path_default = 'refs/remotes'
    _id_attribute_ = 'name'

    def __init__(self, repo: 'Repo', path: PathLike, check_path: bool=False):
        self.repo = repo
        self.path = path

    def __str__(self) -> str:
        return str(self.path)

    def __repr__(self) -> str:
        return '<git.%s "%s">' % (self.__class__.__name__, self.path)

    def __eq__(self, other: object) -> bool:
        if hasattr(other, 'path'):
            other = cast(SymbolicReference, other)
            return self.path == other.path
        return False

    def __ne__(self, other: object) -> bool:
        return not self == other

    def __hash__(self) -> int:
        return hash(self.path)

    @property
    def name(self) -> str:
        """
        :return:
            In case of symbolic references, the shortest assumable name
            is the path itself.
        """
        return str(self.path)

    @property
    def abspath(self) -> PathLike:
        return join_path_native(_git_dir(self.repo, self.path), self.path)

    @classmethod
    def _get_packed_refs_path(cls, repo: 'Repo') -> str:
        return os.path.join(repo.common_dir, 'packed-refs')

    @classmethod
    def _iter_packed_refs(cls, repo: 'Repo') -> Iterator[Tuple[str, str]]:
        """Return an iterator yielding pairs of sha1/path pairs (as strings)
        for the corresponding refs.

        :note: The packed refs file will be kept open as long as we iterate.
        """
        try:
            with open(cls._get_packed_refs_path(repo), 'rt', encoding='UTF-8') as fp:
                for line in fp:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith('#'):
                        if line.startswith('# pack-refs with:') and 'peeled' not in line:
                            raise TypeError('PackingType of packed-Refs not understood: %r' % line)
                        continue
                    if line[0] == '^':
                        continue
                    yield cast(Tuple[str, str], tuple(line.split(' ', 1)))
        except OSError:
            return None

    @classmethod
    def dereference_recursive(cls, repo: 'Repo', ref_path: Union[PathLike, None]) -> str:
        """
        :return: hexsha stored in the reference at the given ref_path, recursively dereferencing all
            intermediate references as required

        :param repo: The repository containing the reference at ref_path
        """
        while True:
            hexsha, ref_path = cls._get_ref_info(repo, ref_path)
            if hexsha is not None:
                return hexsha

    @staticmethod
    def _check_ref_name_valid(ref_path: PathLike) -> None:
        """Check a ref name for validity.

        This is based on the rules described in:
        https://git-scm.com/docs/git-check-ref-format/#_description
        """
        previous: Union[str, None] = None
        one_before_previous: Union[str, None] = None
        for c in str(ref_path):
            if c in ' ~^:?*[\\':
                raise ValueError(f"Invalid reference '{ref_path}': references cannot contain spaces, tildes (~), carets (^), colons (:), question marks (?), asterisks (*), open brackets ([) or backslashes (\\)")
            elif c == '.':
                if previous is None or previous == '/':
                    raise ValueError(f"Invalid reference '{ref_path}': references cannot start with a period (.) or contain '/.'")
                elif previous == '.':
                    raise ValueError(f"Invalid reference '{ref_path}': references cannot contain '..'")
            elif c == '/':
                if previous == '/':
                    raise ValueError(f"Invalid reference '{ref_path}': references cannot contain '//'")
                elif previous is None:
                    raise ValueError(f"Invalid reference '{ref_path}': references cannot start with forward slashes '/'")
            elif c == '{' and previous == '@':
                raise ValueError(f"Invalid reference '{ref_path}': references cannot contain '@{{'")
            elif ord(c) < 32 or ord(c) == 127:
                raise ValueError(f"Invalid reference '{ref_path}': references cannot contain ASCII control characters")
            one_before_previous = previous
            previous = c
        if previous == '.':
            raise ValueError(f"Invalid reference '{ref_path}': references cannot end with a period (.)")
        elif previous == '/':
            raise ValueError(f"Invalid reference '{ref_path}': references cannot end with a forward slash (/)")
        elif previous == '@' and one_before_previous is None:
            raise ValueError(f"Invalid reference '{ref_path}': references cannot be '@'")
        elif any((component.endswith('.lock') for component in str(ref_path).split('/'))):
            raise ValueError(f"Invalid reference '{ref_path}': references cannot have slash-separated components that end with '.lock'")

    @classmethod
    def _get_ref_info_helper(cls, repo: 'Repo', ref_path: Union[PathLike, None]) -> Union[Tuple[str, None], Tuple[None, str]]:
        """
        :return: (str(sha), str(target_ref_path)) if available, the sha the file at
            rela_path points to, or None.

            target_ref_path is the reference we point to, or None.
        """
        if ref_path:
            cls._check_ref_name_valid(ref_path)
        tokens: Union[None, List[str], Tuple[str, str]] = None
        repodir = _git_dir(repo, ref_path)
        try:
            with open(os.path.join(repodir, str(ref_path)), 'rt', encoding='UTF-8') as fp:
                value = fp.read().rstrip()
            tokens = value.split()
            assert len(tokens) != 0
        except OSError:
            for sha, path in cls._iter_packed_refs(repo):
                if path != ref_path:
                    continue
                tokens = (sha, path)
                break
        if tokens is None:
            raise ValueError('Reference at %r does not exist' % ref_path)
        if tokens[0] == 'ref:':
            return (None, tokens[1])
        if repo.re_hexsha_only.match(tokens[0]):
            return (tokens[0], None)
        raise ValueError('Failed to parse reference information from %r' % ref_path)

    @classmethod
    def _get_ref_info(cls, repo: 'Repo', ref_path: Union[PathLike, None]) -> Union[Tuple[str, None], Tuple[None, str]]:
        """
        :return: (str(sha), str(target_ref_path)) if available, the sha the file at
            rela_path points to, or None.

            target_ref_path is the reference we point to, or None.
        """
        return cls._get_ref_info_helper(repo, ref_path)

    def _get_object(self) -> Commit_ish:
        """
        :return:
            The object our ref currently refers to. Refs can be cached, they will
            always point to the actual object as it gets re-created on each query.
        """
        return Object.new_from_sha(self.repo, hex_to_bin(self.dereference_recursive(self.repo, self.path)))

    def _get_commit(self) -> 'Commit':
        """
        :return:
            Commit object we point to. This works for detached and non-detached
            :class:`SymbolicReference` instances. The symbolic reference will be
            dereferenced recursively.
        """
        obj = self._get_object()
        if obj.type == 'tag':
            obj = obj.object
        if obj.type != Commit.type:
            raise TypeError('Symbolic Reference pointed to object %r, commit was required' % obj)
        return obj

    def set_commit(self, commit: Union[Commit, 'SymbolicReference', str], logmsg: Union[str, None]=None) -> 'SymbolicReference':
        """As set_object, but restricts the type of object to be a Commit.

        :raise ValueError: If commit is not a :class:`~git.objects.commit.Commit` object
            or doesn't point to a commit
        :return: self
        """
        invalid_type = False
        if isinstance(commit, Object):
            invalid_type = commit.type != Commit.type
        elif isinstance(commit, SymbolicReference):
            invalid_type = commit.object.type != Commit.type
        else:
            try:
                invalid_type = self.repo.rev_parse(commit).type != Commit.type
            except (BadObject, BadName) as e:
                raise ValueError('Invalid object: %s' % commit) from e
        if invalid_type:
            raise ValueError('Need commit, got %r' % commit)
        self.set_object(commit, logmsg)
        return self

    def set_object(self, object: Union[Commit_ish, 'SymbolicReference', str], logmsg: Union[str, None]=None) -> 'SymbolicReference':
        """Set the object we point to, possibly dereference our symbolic reference first.
        If the reference does not exist, it will be created.

        :param object: A refspec, a :class:`SymbolicReference` or an
            :class:`~git.objects.base.Object` instance.
            :class:`SymbolicReference` instances will be dereferenced beforehand to
            obtain the object they point to.
        :param logmsg: If not None, the message will be used in the reflog entry to be
            written. Otherwise the reflog is not altered.
        :note: Plain :class:`SymbolicReference` instances may not actually point to
            objects by convention.
        :return: self
        """
        if isinstance(object, SymbolicReference):
            object = object.object
        is_detached = True
        try:
            is_detached = self.is_detached
        except ValueError:
            pass
        if is_detached:
            return self.set_reference(object, logmsg)
        return self._get_reference().set_object(object, logmsg)
    commit = property(_get_commit, set_commit, doc='Query or set commits directly')
    object = property(_get_object, set_object, doc='Return the object our ref currently refers to')

    def _get_reference(self) -> 'SymbolicReference':
        """
        :return: Reference Object we point to

        :raise TypeError: If this symbolic reference is detached, hence it doesn't point
            to a reference, but to a commit"""
        sha, target_ref_path = self._get_ref_info(self.repo, self.path)
        if target_ref_path is None:
            raise TypeError('%s is a detached symbolic reference as it points to %r' % (self, sha))
        return self.from_path(self.repo, target_ref_path)

    def set_reference(self, ref: Union[Commit_ish, 'SymbolicReference', str], logmsg: Union[str, None]=None) -> 'SymbolicReference':
        """Set ourselves to the given ref. It will stay a symbol if the ref is a Reference.
        Otherwise an Object, given as Object instance or refspec, is assumed and if valid,
        will be set which effectively detaches the reference if it was a purely
        symbolic one.

        :param ref:
            A :class:`SymbolicReference` instance, an :class:`~git.objects.base.Object`
            instance, or a refspec string. Only if the ref is a
            :class:`SymbolicReference` instance, we will point to it. Everything else is
            dereferenced to obtain the actual object.

        :param logmsg: If set to a string, the message will be used in the reflog.
            Otherwise, a reflog entry is not written for the changed reference.
            The previous commit of the entry will be the commit we point to now.

            See also: :meth:`log_append`

        :return: self

        :note: This symbolic reference will not be dereferenced. For that, see
            :meth:`set_object`.
        """
        write_value = None
        obj = None
        if isinstance(ref, SymbolicReference):
            write_value = 'ref: %s' % ref.path
        elif isinstance(ref, Object):
            obj = ref
            write_value = ref.hexsha
        elif isinstance(ref, str):
            try:
                obj = self.repo.rev_parse(ref + '^{}')
                write_value = obj.hexsha
            except (BadObject, BadName) as e:
                raise ValueError('Could not extract object from %s' % ref) from e
        else:
            raise ValueError('Unrecognized Value: %r' % ref)
        if obj is not None and self._points_to_commits_only and (obj.type != Commit.type):
            raise TypeError('Require commit, got %r' % obj)
        oldbinsha: bytes = b''
        if logmsg is not None:
            try:
                oldbinsha = self.commit.binsha
            except ValueError:
                oldbinsha = Commit.NULL_BIN_SHA
        fpath = self.abspath
        assure_directory_exists(fpath, is_file=True)
        lfd = LockedFD(fpath)
        fd = lfd.open(write=True, stream=True)
        try:
            fd.write(write_value.encode('utf-8') + b'\n')
            lfd.commit()
        except BaseException:
            lfd.rollback()
            raise
        if logmsg is not None:
            self.log_append(oldbinsha, logmsg)
        return self
    reference: Union['Head', 'TagReference', 'RemoteReference', 'Reference']
    reference = property(_get_reference, set_reference, doc='Returns the Reference we point to')
    ref = reference

    def is_valid(self) -> bool:
        """
        :return:
            True if the reference is valid, hence it can be read and points to
            a valid object or reference.
        """
        try:
            self.object
        except (OSError, ValueError):
            return False
        else:
            return True

    @property
    def is_detached(self) -> bool:
        """
        :return:
            True if we are a detached reference, hence we point to a specific commit
            instead to another reference.
        """
        try:
            self.ref
            return False
        except TypeError:
            return True

    def log(self) -> 'RefLog':
        """
        :return: RefLog for this reference. Its last entry reflects the latest change
            applied to this reference.

        .. note:: As the log is parsed every time, its recommended to cache it for use
            instead of calling this method repeatedly. It should be considered read-only.
        """
        return RefLog.from_file(RefLog.path(self))

    def log_append(self, oldbinsha: bytes, message: Union[str, None], newbinsha: Union[bytes, None]=None) -> 'RefLogEntry':
        """Append a logentry to the logfile of this ref.

        :param oldbinsha: Binary sha this ref used to point to.
        :param message: A message describing the change.
        :param newbinsha: The sha the ref points to now. If None, our current commit sha
            will be used.
        :return: The added :class:`~git.refs.log.RefLogEntry` instance.
        """
        try:
            committer_or_reader: Union['Actor', 'GitConfigParser'] = self.commit.committer
        except ValueError:
            committer_or_reader = self.repo.config_reader()
        if newbinsha is None:
            newbinsha = self.commit.binsha
        if message is None:
            message = ''
        return RefLog.append_entry(committer_or_reader, RefLog.path(self), oldbinsha, newbinsha, message)

    def log_entry(self, index: int) -> 'RefLogEntry':
        """
        :return: RefLogEntry at the given index

        :param index: Python list compatible positive or negative index

        .. note:: This method must read part of the reflog during execution, hence
            it should be used sparingly, or only if you need just one index.
            In that case, it will be faster than the :meth:`log` method.
        """
        return RefLog.entry_at(RefLog.path(self), index)

    @classmethod
    def to_full_path(cls, path: Union[PathLike, 'SymbolicReference']) -> PathLike:
        """
        :return: string with a full repository-relative path which can be used to initialize
            a Reference instance, for instance by using
            :meth:`Reference.from_path <git.refs.reference.Reference.from_path>`.
        """
        if isinstance(path, SymbolicReference):
            path = path.path
        full_ref_path = path
        if not cls._common_path_default:
            return full_ref_path
        if not str(path).startswith(cls._common_path_default + '/'):
            full_ref_path = '%s/%s' % (cls._common_path_default, path)
        return full_ref_path

    @classmethod
    def delete(cls, repo: 'Repo', path: PathLike) -> None:
        """Delete the reference at the given path.

        :param repo:
            Repository to delete the reference from.

        :param path:
            Short or full path pointing to the reference, e.g. ``refs/myreference``
            or just ``myreference``, hence ``refs/`` is implied.
            Alternatively the symbolic reference to be deleted.
        """
        full_ref_path = cls.to_full_path(path)
        abs_path = os.path.join(repo.common_dir, full_ref_path)
        if os.path.exists(abs_path):
            os.remove(abs_path)
        else:
            pack_file_path = cls._get_packed_refs_path(repo)
            try:
                with open(pack_file_path, 'rb') as reader:
                    new_lines = []
                    made_change = False
                    dropped_last_line = False
                    for line_bytes in reader:
                        line = line_bytes.decode(defenc)
                        _, _, line_ref = line.partition(' ')
                        line_ref = line_ref.strip()
                        if (line.startswith('#') or full_ref_path != line_ref) and (not dropped_last_line or (dropped_last_line and (not line.startswith('^')))):
                            new_lines.append(line)
                            dropped_last_line = False
                            continue
                        made_change = True
                        dropped_last_line = True
                if made_change:
                    with open(pack_file_path, 'wb') as fd:
                        fd.writelines((line.encode(defenc) for line in new_lines))
            except OSError:
                pass
        reflog_path = RefLog.path(cls(repo, full_ref_path))
        if os.path.isfile(reflog_path):
            os.remove(reflog_path)

    @classmethod
    def _create(cls: Type[T_References], repo: 'Repo', path: PathLike, resolve: bool, reference: Union['SymbolicReference', str], force: bool, logmsg: Union[str, None]=None) -> T_References:
        """Internal method used to create a new symbolic reference.

        If `resolve` is False, the reference will be taken as is, creating
        a proper symbolic reference. Otherwise it will be resolved to the
        corresponding object and a detached symbolic reference will be created
        instead.
        """
        git_dir = _git_dir(repo, path)
        full_ref_path = cls.to_full_path(path)
        abs_ref_path = os.path.join(git_dir, full_ref_path)
        target = reference
        if resolve:
            target = repo.rev_parse(str(reference))
        if not force and os.path.isfile(abs_ref_path):
            target_data = str(target)
            if isinstance(target, SymbolicReference):
                target_data = str(target.path)
            if not resolve:
                target_data = 'ref: ' + target_data
            with open(abs_ref_path, 'rb') as fd:
                existing_data = fd.read().decode(defenc).strip()
            if existing_data != target_data:
                raise OSError('Reference at %r does already exist, pointing to %r, requested was %r' % (full_ref_path, existing_data, target_data))
        ref = cls(repo, full_ref_path)
        ref.set_reference(target, logmsg)
        return ref

    @classmethod
    def create(cls: Type[T_References], repo: 'Repo', path: PathLike, reference: Union['SymbolicReference', str]='HEAD', logmsg: Union[str, None]=None, force: bool=False, **kwargs: Any) -> T_References:
        """Create a new symbolic reference: a reference pointing to another reference.

        :param repo:
            Repository to create the reference in.

        :param path:
            Full path at which the new symbolic reference is supposed to be
            created at, e.g. ``NEW_HEAD`` or ``symrefs/my_new_symref``.

        :param reference:
            The reference which the new symbolic reference should point to.
            If it is a commit-ish, the symbolic ref will be detached.

        :param force:
            If True, force creation even if a symbolic reference with that name already exists.
            Raise :class:`OSError` otherwise.

        :param logmsg:
            If not None, the message to append to the reflog. Otherwise no reflog
            entry is written.

        :return: Newly created symbolic Reference

        :raise OSError:
            If a (Symbolic)Reference with the same name but different contents
            already exists.

        :note: This does not alter the current HEAD, index or working tree.
        """
        return cls._create(repo, path, cls._resolve_ref_on_create, reference, force, logmsg)

    def rename(self, new_path: PathLike, force: bool=False) -> 'SymbolicReference':
        """Rename self to a new path.

        :param new_path:
            Either a simple name or a full path, e.g. ``new_name`` or ``features/new_name``.
            The prefix ``refs/`` is implied for references and will be set as needed.
            In case this is a symbolic ref, there is no implied prefix.

        :param force:
            If True, the rename will succeed even if a head with the target name
            already exists. It will be overwritten in that case.

        :return: self

        :raise OSError: If a file at path but with different contents already exists.
        """
        new_path = self.to_full_path(new_path)
        if self.path == new_path:
            return self
        new_abs_path = os.path.join(_git_dir(self.repo, new_path), new_path)
        cur_abs_path = os.path.join(_git_dir(self.repo, self.path), self.path)
        if os.path.isfile(new_abs_path):
            if not force:
                with open(new_abs_path, 'rb') as fd1:
                    f1 = fd1.read().strip()
                with open(cur_abs_path, 'rb') as fd2:
                    f2 = fd2.read().strip()
                if f1 != f2:
                    raise OSError('File at path %r already exists' % new_abs_path)
            os.remove(new_abs_path)
        dname = os.path.dirname(new_abs_path)
        if not os.path.isdir(dname):
            os.makedirs(dname)
        os.rename(cur_abs_path, new_abs_path)
        self.path = new_path
        return self

    @classmethod
    def _iter_items(cls: Type[T_References], repo: 'Repo', common_path: Union[PathLike, None]=None) -> Iterator[T_References]:
        if common_path is None:
            common_path = cls._common_path_default
        rela_paths = set()
        for root, dirs, files in os.walk(join_path_native(repo.common_dir, common_path)):
            if 'refs' not in root.split(os.sep):
                refs_id = [d for d in dirs if d == 'refs']
                if refs_id:
                    dirs[0:] = ['refs']
            for f in files:
                if f == 'packed-refs':
                    continue
                abs_path = to_native_path_linux(join_path(root, f))
                rela_paths.add(abs_path.replace(to_native_path_linux(repo.common_dir) + '/', ''))
        for _sha, rela_path in cls._iter_packed_refs(repo):
            if rela_path.startswith(str(common_path)):
                rela_paths.add(rela_path)
        for path in sorted(rela_paths):
            try:
                yield cls.from_path(repo, path)
            except ValueError:
                continue

    @classmethod
    def iter_items(cls: Type[T_References], repo: 'Repo', common_path: Union[PathLike, None]=None, *args: Any, **kwargs: Any) -> Iterator[T_References]:
        """Find all refs in the repository.

        :param repo: is the Repo

        :param common_path:
            Optional keyword argument to the path which is to be shared by all returned
            Ref objects.
            Defaults to class specific portion if None, ensuring that only refs suitable
            for the actual class are returned.

        :return:
            A list of :class:`SymbolicReference`, each guaranteed to be a symbolic ref
            which is not detached and pointing to a valid ref.

            The list is lexicographically sorted. The returned objects are instances of
            concrete subclasses, such as :class:`~git.refs.head.Head` or
            :class:`~git.refs.tag.TagReference`.
        """
        return (r for r in cls._iter_items(repo, common_path) if r.__class__ is SymbolicReference or not r.is_detached)

    @classmethod
    def from_path(cls: Type[T_References], repo: 'Repo', path: PathLike) -> T_References:
        """
        Make a symbolic reference from a path.

        :param path: Full ``.git``-directory-relative path name to the Reference to
            instantiate.

        :note: Use :meth:`to_full_path` if you only have a partial path of a known
            Reference type.

        :return:
            Instance of type :class:`~git.refs.reference.Reference`,
            :class:`~git.refs.head.Head`, or :class:`~git.refs.tag.Tag`, depending on
            the given path.
        """
        if not path:
            raise ValueError('Cannot create Reference from %r' % path)
        from . import HEAD, Head, RemoteReference, TagReference, Reference
        for ref_type in (HEAD, Head, RemoteReference, TagReference, Reference, SymbolicReference):
            try:
                instance: T_References
                instance = ref_type(repo, path)
                if instance.__class__ is SymbolicReference and instance.is_detached:
                    raise ValueError('SymbolicRef was detached, we drop it')
                else:
                    return instance
            except ValueError:
                pass
        raise ValueError('Could not find reference type suitable to handle path %r' % path)

    def is_remote(self) -> bool:
        """:return: True if this symbolic reference points to a remote branch"""
        return str(self.path).startswith(self._remote_common_path_default + '/')