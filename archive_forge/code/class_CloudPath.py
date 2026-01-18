import abc
from collections import defaultdict
import collections.abc
from contextlib import contextmanager
import os
from pathlib import (  # type: ignore
import shutil
import sys
from typing import (
from urllib.parse import urlparse
from warnings import warn
from cloudpathlib.enums import FileCacheMode
from . import anypath
from .exceptions import (
class CloudPath(metaclass=CloudPathMeta):
    """Base class for cloud storage file URIs, in the style of the Python standard library's
    [`pathlib` module](https://docs.python.org/3/library/pathlib.html). Instances represent a path
    in cloud storage with filesystem path semantics, and convenient methods allow for basic
    operations like joining, reading, writing, iterating over contents, etc. `CloudPath` almost
    entirely mimics the [`pathlib.Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)
    interface, so most familiar properties and methods should be available and behave in the
    expected way.

    Analogous to the way `pathlib.Path` works, instantiating `CloudPath` will instead create an
    instance of an appropriate subclass that implements a particular cloud storage service, such as
    [`S3Path`](../s3path). This dispatching behavior is based on the URI scheme part of a cloud
    storage URI (e.g., `"s3://"`).
    """
    _cloud_meta: CloudImplementation
    cloud_prefix: str

    def __init__(self, cloud_path: Union[str, Self, 'CloudPath'], client: Optional['Client']=None) -> None:
        self._handle: Optional[IO] = None
        self.is_valid_cloudpath(cloud_path, raise_on_error=True)
        self._str = str(cloud_path)
        self._url = urlparse(self._str)
        self._path = PurePosixPath(f'/{self._no_prefix}')
        if client is None:
            if isinstance(cloud_path, CloudPath):
                client = cloud_path.client
            else:
                client = self._cloud_meta.client_class.get_default_client()
        if not isinstance(client, self._cloud_meta.client_class):
            raise ClientMismatchError(f'Client of type [{client.__class__}] is not valid for cloud path of type [{self.__class__}]; must be instance of [{self._cloud_meta.client_class}], or None to use default client for this cloud path class.')
        self.client: Client = client
        self._dirty = False

    def __del__(self) -> None:
        if self._handle is not None:
            self._handle.close()
        if hasattr(self, 'client') and self.client.file_cache_mode == FileCacheMode.cloudpath_object:
            self.clear_cache()

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        del state['client']
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        client = self._cloud_meta.client_class.get_default_client()
        state['client'] = client
        self.__dict__.update(state)

    @property
    def _no_prefix(self) -> str:
        return self._str[len(self.cloud_prefix):]

    @property
    def _no_prefix_no_drive(self) -> str:
        return self._str[len(self.cloud_prefix) + len(self.drive):]

    @overload
    @classmethod
    def is_valid_cloudpath(cls, path: 'CloudPath', raise_on_error: bool=...) -> TypeGuard[Self]:
        ...

    @overload
    @classmethod
    def is_valid_cloudpath(cls, path: str, raise_on_error: bool=...) -> bool:
        ...

    @classmethod
    def is_valid_cloudpath(cls, path: Union[str, 'CloudPath'], raise_on_error: bool=False) -> Union[bool, TypeGuard[Self]]:
        valid = str(path).lower().startswith(cls.cloud_prefix.lower())
        if raise_on_error and (not valid):
            raise InvalidPrefixError(f"'{path}' is not a valid path since it does not start with '{cls.cloud_prefix}'")
        return valid

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self}')"

    def __str__(self) -> str:
        return self._str

    def __hash__(self) -> int:
        return hash((type(self).__name__, str(self)))

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, type(self)) and str(self) == str(other)

    def __fspath__(self) -> str:
        if self.is_file():
            self._refresh_cache(force_overwrite_from_cloud=False)
        return str(self._local)

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.parts < other.parts

    def __le__(self, other: Any) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.parts <= other.parts

    def __gt__(self, other: Any) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.parts > other.parts

    def __ge__(self, other: Any) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.parts >= other.parts

    @property
    @abc.abstractmethod
    def drive(self) -> str:
        """For example "bucket" on S3 or "container" on Azure; needs to be defined for each class"""
        pass

    @abc.abstractmethod
    def is_dir(self) -> bool:
        """Should be implemented without requiring a dir is downloaded"""
        pass

    @abc.abstractmethod
    def is_file(self) -> bool:
        """Should be implemented without requiring that the file is downloaded"""
        pass

    @abc.abstractmethod
    def mkdir(self, parents: bool=False, exist_ok: bool=False) -> None:
        """Should be implemented using the client API without requiring a dir is downloaded"""
        pass

    @abc.abstractmethod
    def touch(self, exist_ok: bool=True) -> None:
        """Should be implemented using the client API to create and update modified time"""
        pass

    def __rtruediv__(self, other: Any) -> None:
        raise ValueError("Cannot change a cloud path's root since all paths are absolute; create a new path instead.")

    @property
    def anchor(self) -> str:
        return self.cloud_prefix

    def as_uri(self) -> str:
        return str(self)

    def exists(self) -> bool:
        return self.client._exists(self)

    @property
    def fspath(self) -> str:
        return self.__fspath__()

    def _glob_checks(self, pattern: str) -> None:
        if '..' in pattern:
            raise CloudPathNotImplementedError("Relative paths with '..' not supported in glob patterns.")
        if pattern.startswith(self.cloud_prefix) or pattern.startswith('/'):
            raise CloudPathNotImplementedError('Non-relative patterns are unsupported')
        if self.drive == '':
            raise CloudPathNotImplementedError(".glob is only supported within a bucket or container; you can use `.iterdir` to list buckets; for example, CloudPath('s3://').iterdir()")

    def _build_subtree(self, recursive):
        Tree: Callable = lambda: defaultdict(Tree)

        def _build_tree(trunk, branch, nodes, is_dir):
            """Utility to build a tree from nested defaultdicts with a generator
            of nodes (parts) of a path."""
            next_branch = next(nodes, None)
            if next_branch is None:
                trunk[branch] = Tree() if is_dir else None
            else:
                _build_tree(trunk[branch], next_branch, nodes, is_dir)
        file_tree = Tree()
        for f, is_dir in self.client._list_dir(self, recursive=recursive):
            parts = str(f.relative_to(self)).split('/')
            if len(parts) == 1 and parts[0] == '.':
                continue
            nodes = (p for p in parts)
            _build_tree(file_tree, next(nodes, None), nodes, is_dir)
        return dict(file_tree)

    def _glob(self, selector, recursive: bool) -> Generator[Self, None, None]:
        file_tree = self._build_subtree(recursive)
        root = _CloudPathSelectable(self.name, [], file_tree)
        for p in selector.select_from(root):
            yield (self / str(p)[len(self.name) + 1:])

    def glob(self, pattern: str, case_sensitive: Optional[bool]=None) -> Generator[Self, None, None]:
        self._glob_checks(pattern)
        pattern_parts = PurePosixPath(pattern).parts
        selector = _make_selector(tuple(pattern_parts), _posix_flavour, case_sensitive=case_sensitive)
        yield from self._glob(selector, '/' in pattern or '**' in pattern)

    def rglob(self, pattern: str, case_sensitive: Optional[bool]=None) -> Generator[Self, None, None]:
        self._glob_checks(pattern)
        pattern_parts = PurePosixPath(pattern).parts
        selector = _make_selector(('**',) + tuple(pattern_parts), _posix_flavour, case_sensitive=case_sensitive)
        yield from self._glob(selector, True)

    def iterdir(self) -> Generator[Self, None, None]:
        for f, _ in self.client._list_dir(self, recursive=False):
            if f != self:
                yield f

    @staticmethod
    def _walk_results_from_tree(root, tree, top_down=True):
        """Utility to yield tuples in the form expected by `.walk` from the file
        tree constructed by `_build_substree`.
        """
        dirs = []
        files = []
        for item, branch in tree.items():
            files.append(item) if branch is None else dirs.append(item)
        if top_down:
            yield (root, dirs, files)
        for dir in dirs:
            yield from CloudPath._walk_results_from_tree(root / dir, tree[dir], top_down=top_down)
        if not top_down:
            yield (root, dirs, files)

    def walk(self, top_down: bool=True, on_error: Optional[Callable]=None, follow_symlinks: bool=False) -> Generator[Tuple[Self, List[str], List[str]], None, None]:
        try:
            file_tree = self._build_subtree(recursive=True)
            yield from self._walk_results_from_tree(self, file_tree, top_down=top_down)
        except Exception as e:
            if on_error is not None:
                on_error(e)
            else:
                raise

    def open(self, mode: str='r', buffering: int=-1, encoding: Optional[str]=None, errors: Optional[str]=None, newline: Optional[str]=None, force_overwrite_from_cloud: bool=False, force_overwrite_to_cloud: bool=False) -> IO[Any]:
        if self.exists() and (not self.is_file()):
            raise CloudPathIsADirectoryError(f'Cannot open directory, only files. Tried to open ({self})')
        if mode == 'x' and self.exists():
            raise CloudPathFileExistsError(f'Cannot open existing file ({self}) for creation.')
        self._refresh_cache(force_overwrite_from_cloud=force_overwrite_from_cloud)
        if not self._local.exists():
            self._local.parent.mkdir(parents=True, exist_ok=True)
            original_mtime = 0.0
        else:
            original_mtime = self._local.stat().st_mtime
        buffer = self._local.open(mode=mode, buffering=buffering, encoding=encoding, errors=errors, newline=newline)
        if any((m in mode for m in ('w', '+', 'x', 'a'))):
            wrapped_close = buffer.close

            def _patched_close_upload(*args, **kwargs) -> None:
                wrapped_close(*args, **kwargs)
                if not self._dirty:
                    return
                if self._local.stat().st_mtime < original_mtime:
                    new_mtime = original_mtime + 1
                    os.utime(self._local, times=(new_mtime, new_mtime))
                self._upload_local_to_cloud(force_overwrite_to_cloud=force_overwrite_to_cloud)
                self._dirty = False
            buffer.close = _patched_close_upload
            self._handle = buffer
            self._dirty = True
        if self.client.file_cache_mode == FileCacheMode.close_file:
            wrapped_close_for_cache = buffer.close

            def _patched_close_empty_cache(*args, **kwargs):
                wrapped_close_for_cache(*args, **kwargs)
                self.clear_cache()
            buffer.close = _patched_close_empty_cache
        return buffer

    def replace(self, target: Self) -> Self:
        if type(self) is not type(target):
            raise TypeError(f'The target based to rename must be an instantiated class of type: {type(self)}')
        if self.is_dir():
            raise CloudPathIsADirectoryError(f'Path {self} is a directory; rename/replace the files recursively.')
        if target == self:
            return self
        if target.exists():
            target.unlink()
        self.client._move_file(self, target)
        return target

    def rename(self, target: Self) -> Self:
        return self.replace(target)

    def rmdir(self) -> None:
        if self.is_file():
            raise CloudPathNotADirectoryError(f'Path {self} is a file; call unlink instead of rmdir.')
        try:
            next(self.iterdir())
            raise DirectoryNotEmptyError(f"Directory not empty: '{self}'. Use rmtree to delete recursively.")
        except StopIteration:
            pass
        self.client._remove(self)

    def samefile(self, other_path: Union[str, os.PathLike]) -> bool:
        return self == other_path

    def unlink(self, missing_ok: bool=True) -> None:
        if self.is_dir():
            raise CloudPathIsADirectoryError(f'Path {self} is a directory; call rmdir instead of unlink.')
        self.client._remove(self, missing_ok)

    def write_bytes(self, data: bytes) -> int:
        """Open the file in bytes mode, write to it, and close the file.

        NOTE: vendored from pathlib since we override open
        https://github.com/python/cpython/blob/3.8/Lib/pathlib.py#L1235-L1242
        """
        view = memoryview(data)
        with self.open(mode='wb') as f:
            return f.write(view)

    def write_text(self, data: str, encoding: Optional[str]=None, errors: Optional[str]=None, newline: Optional[str]=None) -> int:
        """Open the file in text mode, write to it, and close the file.

        NOTE: vendored from pathlib since we override open
        https://github.com/python/cpython/blob/3.10/Lib/pathlib.py#L1146-L1155
        """
        if not isinstance(data, str):
            raise TypeError('data must be str, not %s' % data.__class__.__name__)
        with self.open(mode='w', encoding=encoding, errors=errors, newline=newline) as f:
            return f.write(data)

    def read_bytes(self) -> bytes:
        with self.open(mode='rb') as f:
            return f.read()

    def read_text(self, encoding: Optional[str]=None, errors: Optional[str]=None) -> str:
        with self.open(mode='r', encoding=encoding, errors=errors) as f:
            return f.read()

    def is_junction(self):
        return False

    def _dispatch_to_path(self, func: str, *args, **kwargs) -> Any:
        """Some functions we can just dispatch to the pathlib version
        We want to do this explicitly so we don't have to support all
        of pathlib and subclasses can override individually if necessary.
        """
        path_version = self._path.__getattribute__(func)
        if callable(path_version):
            path_version = path_version(*args, **kwargs)
        if isinstance(path_version, PurePosixPath):
            path_version = _resolve(path_version)
            return self._new_cloudpath(path_version)
        if isinstance(path_version, collections.abc.Sequence) and len(path_version) > 0 and isinstance(path_version[0], PurePosixPath):
            sequence_class = type(path_version) if not isinstance(path_version, _PathParents) else tuple
            return sequence_class((self._new_cloudpath(_resolve(p)) for p in path_version if _resolve(p) != p.root))
        else:
            return path_version

    def __truediv__(self, other: Union[str, PurePosixPath]) -> Self:
        if not isinstance(other, (str, PurePosixPath)):
            raise TypeError(f'Can only join path {repr(self)} with strings or posix paths.')
        return self._dispatch_to_path('__truediv__', other)

    def joinpath(self, *pathsegments: Union[str, os.PathLike]) -> Self:
        return self._dispatch_to_path('joinpath', *pathsegments)

    def absolute(self) -> Self:
        return self

    def is_absolute(self) -> bool:
        return True

    def resolve(self, strict: bool=False) -> Self:
        return self

    def relative_to(self, other: Self, walk_up: bool=False) -> PurePosixPath:
        if not isinstance(other, CloudPath):
            raise ValueError(f'{self} is a cloud path, but {other} is not')
        if self.cloud_prefix != other.cloud_prefix:
            raise ValueError(f'{self} is a {self.cloud_prefix} path, but {other} is a {other.cloud_prefix} path')
        kwargs = dict(walk_up=walk_up)
        if sys.version_info < (3, 12):
            kwargs.pop('walk_up')
        return self._path.relative_to(other._path, **kwargs)

    def is_relative_to(self, other: Self) -> bool:
        try:
            self.relative_to(other)
            return True
        except ValueError:
            return False

    @property
    def name(self) -> str:
        return self._dispatch_to_path('name')

    def match(self, path_pattern: str, case_sensitive: Optional[bool]=None) -> bool:
        if path_pattern.startswith(self.anchor + self.drive + '/'):
            path_pattern = path_pattern[len(self.anchor + self.drive + '/'):]
        kwargs = dict(case_sensitive=case_sensitive)
        if sys.version_info < (3, 12):
            kwargs.pop('case_sensitive')
        return self._dispatch_to_path('match', path_pattern, **kwargs)

    @property
    def parent(self) -> Self:
        return self._dispatch_to_path('parent')

    @property
    def parents(self) -> Sequence[Self]:
        return self._dispatch_to_path('parents')

    @property
    def parts(self) -> Tuple[str, ...]:
        parts = self._dispatch_to_path('parts')
        if parts[0] == '/':
            parts = parts[1:]
        return (self.anchor, *parts)

    @property
    def stem(self) -> str:
        return self._dispatch_to_path('stem')

    @property
    def suffix(self) -> str:
        return self._dispatch_to_path('suffix')

    @property
    def suffixes(self) -> List[str]:
        return self._dispatch_to_path('suffixes')

    def with_stem(self, stem: str) -> Self:
        try:
            return self._dispatch_to_path('with_stem', stem)
        except AttributeError:
            return self.with_name(stem + self.suffix)

    def with_name(self, name: str) -> Self:
        return self._dispatch_to_path('with_name', name)

    def with_segments(self, *pathsegments) -> Self:
        """Create a new CloudPath with the same client out of the given segments.
        The first segment will be interpreted as the bucket/container name.
        """
        return self._new_cloudpath('/'.join(pathsegments))

    def with_suffix(self, suffix: str) -> Self:
        return self._dispatch_to_path('with_suffix', suffix)

    def _dispatch_to_local_cache_path(self, func: str, *args, **kwargs) -> Any:
        self._refresh_cache()
        path_version = self._local.__getattribute__(func)
        if callable(path_version):
            path_version = path_version(*args, **kwargs)
        if isinstance(path_version, (PosixPath, WindowsPath)):
            path_version = path_version.resolve()
            return self._new_cloudpath(path_version)
        else:
            return path_version

    def stat(self, follow_symlinks: bool=True) -> os.stat_result:
        """Note: for many clients, we may want to override so we don't incur
        network costs since many of these properties are available as
        API calls.
        """
        warn(f'stat not implemented as API call for {self.__class__} so file must be downloaded to calculate stats; this may take a long time depending on filesize')
        return self._dispatch_to_local_cache_path('stat', follow_symlinks=follow_symlinks)

    def download_to(self, destination: Union[str, os.PathLike]) -> Path:
        destination = Path(destination)
        if self.is_file():
            if destination.is_dir():
                destination = destination / self.name
            return self.client._download_file(self, destination)
        else:
            destination.mkdir(exist_ok=True)
            for f in self.iterdir():
                rel = str(self)
                if not rel.endswith('/'):
                    rel = rel + '/'
                rel_dest = str(f)[len(rel):]
                f.download_to(destination / rel_dest)
            return destination

    def rmtree(self) -> None:
        """Delete an entire directory tree."""
        if self.is_file():
            raise CloudPathNotADirectoryError(f'Path {self} is a file; call unlink instead of rmtree.')
        self.client._remove(self)

    def upload_from(self, source: Union[str, os.PathLike], force_overwrite_to_cloud: bool=False) -> Self:
        """Upload a file or directory to the cloud path."""
        source = Path(source)
        if source.is_dir():
            for p in source.iterdir():
                (self / p.name).upload_from(p, force_overwrite_to_cloud=force_overwrite_to_cloud)
            return self
        else:
            if self.exists() and self.is_dir():
                dst = self / source.name
            else:
                dst = self
            dst._upload_file_to_cloud(source, force_overwrite_to_cloud=force_overwrite_to_cloud)
            return dst

    @overload
    def copy(self, destination: Self, force_overwrite_to_cloud: bool=False) -> Self:
        ...

    @overload
    def copy(self, destination: Path, force_overwrite_to_cloud: bool=False) -> Path:
        ...

    @overload
    def copy(self, destination: str, force_overwrite_to_cloud: bool=False) -> Union[Path, 'CloudPath']:
        ...

    def copy(self, destination, force_overwrite_to_cloud=False):
        """Copy self to destination folder of file, if self is a file."""
        if not self.exists() or not self.is_file():
            raise ValueError(f'Path {self} should be a file. To copy a directory tree use the method copytree.')
        if isinstance(destination, (str, os.PathLike)):
            destination = anypath.to_anypath(destination)
        if not isinstance(destination, CloudPath):
            return self.download_to(destination)
        if self.client is destination.client:
            if destination.exists() and destination.is_dir():
                destination = destination / self.name
            if not force_overwrite_to_cloud and destination.exists() and (destination.stat().st_mtime >= self.stat().st_mtime):
                raise OverwriteNewerCloudError(f'File ({destination}) is newer than ({self}). To overwrite pass `force_overwrite_to_cloud=True`.')
            return self.client._move_file(self, destination, remove_src=False)
        elif not destination.exists() or destination.is_file():
            return destination.upload_from(self.fspath, force_overwrite_to_cloud=force_overwrite_to_cloud)
        else:
            return (destination / self.name).upload_from(self.fspath, force_overwrite_to_cloud=force_overwrite_to_cloud)

    @overload
    def copytree(self, destination: Self, force_overwrite_to_cloud: bool=False, ignore: Optional[Callable[[str, Iterable[str]], Container[str]]]=None) -> Self:
        ...

    @overload
    def copytree(self, destination: Path, force_overwrite_to_cloud: bool=False, ignore: Optional[Callable[[str, Iterable[str]], Container[str]]]=None) -> Path:
        ...

    @overload
    def copytree(self, destination: str, force_overwrite_to_cloud: bool=False, ignore: Optional[Callable[[str, Iterable[str]], Container[str]]]=None) -> Union[Path, 'CloudPath']:
        ...

    def copytree(self, destination, force_overwrite_to_cloud=False, ignore=None):
        """Copy self to a directory, if self is a directory."""
        if not self.is_dir():
            raise CloudPathNotADirectoryError(f'Origin path {self} must be a directory. To copy a single file use the method copy.')
        if isinstance(destination, (str, os.PathLike)):
            destination = anypath.to_anypath(destination)
        if destination.exists() and destination.is_file():
            raise CloudPathFileExistsError(f'Destination path {destination} of copytree must be a directory.')
        contents = list(self.iterdir())
        if ignore is not None:
            ignored_names = ignore(self._no_prefix_no_drive, [x.name for x in contents])
        else:
            ignored_names = set()
        destination.mkdir(parents=True, exist_ok=True)
        for subpath in contents:
            if subpath.name in ignored_names:
                continue
            if subpath.is_file():
                subpath.copy(destination / subpath.name, force_overwrite_to_cloud=force_overwrite_to_cloud)
            elif subpath.is_dir():
                subpath.copytree(destination / subpath.name, force_overwrite_to_cloud=force_overwrite_to_cloud, ignore=ignore)
        return destination

    def clear_cache(self):
        """Removes cache if it exists"""
        if self._local.exists():
            if self._local.is_file():
                self._local.unlink()
            else:
                shutil.rmtree(self._local)

    @property
    def _local(self) -> Path:
        """Cached local version of the file."""
        return self.client._local_cache_dir / self._no_prefix

    def _new_cloudpath(self, path: Union[str, os.PathLike]) -> Self:
        """Use the scheme, client, cache dir of this cloudpath to instantiate
        a new cloudpath of the same type with the path passed.

        Used to make results of iterdir and joins have a unified client + cache.
        """
        path = str(path)
        if path.startswith('/'):
            path = path[1:]
        if not path.startswith(self.cloud_prefix):
            path = f'{self.cloud_prefix}{path}'
        return self.client.CloudPath(path)

    def _refresh_cache(self, force_overwrite_from_cloud: bool=False) -> None:
        try:
            stats = self.stat()
        except NoStatError:
            return
        if not self._local.exists() or self._local.stat().st_mtime < stats.st_mtime or force_overwrite_from_cloud:
            self._local.parent.mkdir(parents=True, exist_ok=True)
            self.download_to(self._local)
            os.utime(self._local, times=(stats.st_mtime, stats.st_mtime))
        if self._dirty:
            raise OverwriteDirtyFileError(f'Local file ({self._local}) for cloud path ({self}) has been changed by your code, but is being requested for download from cloud. Either (1) push your changes to the cloud, (2) remove the local file, or (3) pass `force_overwrite_from_cloud=True` to overwrite.')
        if self._local.stat().st_mtime > stats.st_mtime:
            raise OverwriteNewerLocalError(f'Local file ({self._local}) for cloud path ({self}) is newer on disk, but is being requested for download from cloud. Either (1) push your changes to the cloud, (2) remove the local file, or (3) pass `force_overwrite_from_cloud=True` to overwrite.')

    def _upload_local_to_cloud(self, force_overwrite_to_cloud: bool=False) -> Self:
        """Uploads cache file at self._local to the cloud"""
        if self._local.is_dir():
            raise ValueError('Only individual files can be uploaded to the cloud')
        uploaded = self._upload_file_to_cloud(self._local, force_overwrite_to_cloud=force_overwrite_to_cloud)
        stats = self.stat()
        os.utime(self._local, times=(stats.st_mtime, stats.st_mtime))
        self._dirty = False
        self._handle = None
        return uploaded

    def _upload_file_to_cloud(self, local_path: Path, force_overwrite_to_cloud: bool=False) -> Self:
        """Uploads file at `local_path` to the cloud if there is not a newer file
        already there.
        """
        try:
            stats = self.stat()
        except NoStatError:
            stats = None
        if not stats or local_path.stat().st_mtime > stats.st_mtime or force_overwrite_to_cloud:
            self.client._upload_file(local_path, self)
            return self
        raise OverwriteNewerCloudError(f'Local file ({self._local}) for cloud path ({self}) is newer in the cloud disk, but is being requested to be uploaded to the cloud. Either (1) redownload changes from the cloud or (2) pass `force_overwrite_to_cloud=True` to overwrite.')

    @classmethod
    def __get_pydantic_core_schema__(cls, _source_type: Any, _handler):
        """Pydantic special method. See
        https://docs.pydantic.dev/2.0/usage/types/custom/"""
        try:
            from pydantic_core import core_schema
            return core_schema.no_info_after_validator_function(cls.validate, core_schema.any_schema())
        except ImportError:
            return None

    @classmethod
    def validate(cls, v: str) -> Self:
        """Used as a Pydantic validator. See
        https://docs.pydantic.dev/2.0/usage/types/custom/"""
        return cls(v)

    @classmethod
    def __get_validators__(cls) -> Generator[Callable[[Any], Self], None, None]:
        """Pydantic special method. See
        https://pydantic-docs.helpmanual.io/usage/types/#custom-data-types"""
        yield cls._validate

    @classmethod
    def _validate(cls, value: Any) -> Self:
        """Used as a Pydantic validator. See
        https://pydantic-docs.helpmanual.io/usage/types/#custom-data-types"""
        return cls(value)