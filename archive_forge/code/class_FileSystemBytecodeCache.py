import errno
import fnmatch
import marshal
import os
import pickle
import stat
import sys
import tempfile
import typing as t
from hashlib import sha1
from io import BytesIO
from types import CodeType
class FileSystemBytecodeCache(BytecodeCache):
    """A bytecode cache that stores bytecode on the filesystem.  It accepts
    two arguments: The directory where the cache items are stored and a
    pattern string that is used to build the filename.

    If no directory is specified a default cache directory is selected.  On
    Windows the user's temp directory is used, on UNIX systems a directory
    is created for the user in the system temp directory.

    The pattern can be used to have multiple separate caches operate on the
    same directory.  The default pattern is ``'__jinja2_%s.cache'``.  ``%s``
    is replaced with the cache key.

    >>> bcc = FileSystemBytecodeCache('/tmp/jinja_cache', '%s.cache')

    This bytecode cache supports clearing of the cache using the clear method.
    """

    def __init__(self, directory: t.Optional[str]=None, pattern: str='__jinja2_%s.cache') -> None:
        if directory is None:
            directory = self._get_default_cache_dir()
        self.directory = directory
        self.pattern = pattern

    def _get_default_cache_dir(self) -> str:

        def _unsafe_dir() -> 'te.NoReturn':
            raise RuntimeError('Cannot determine safe temp directory.  You need to explicitly provide one.')
        tmpdir = tempfile.gettempdir()
        if os.name == 'nt':
            return tmpdir
        if not hasattr(os, 'getuid'):
            _unsafe_dir()
        dirname = f'_jinja2-cache-{os.getuid()}'
        actual_dir = os.path.join(tmpdir, dirname)
        try:
            os.mkdir(actual_dir, stat.S_IRWXU)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        try:
            os.chmod(actual_dir, stat.S_IRWXU)
            actual_dir_stat = os.lstat(actual_dir)
            if actual_dir_stat.st_uid != os.getuid() or not stat.S_ISDIR(actual_dir_stat.st_mode) or stat.S_IMODE(actual_dir_stat.st_mode) != stat.S_IRWXU:
                _unsafe_dir()
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        actual_dir_stat = os.lstat(actual_dir)
        if actual_dir_stat.st_uid != os.getuid() or not stat.S_ISDIR(actual_dir_stat.st_mode) or stat.S_IMODE(actual_dir_stat.st_mode) != stat.S_IRWXU:
            _unsafe_dir()
        return actual_dir

    def _get_cache_filename(self, bucket: Bucket) -> str:
        return os.path.join(self.directory, self.pattern % (bucket.key,))

    def load_bytecode(self, bucket: Bucket) -> None:
        filename = self._get_cache_filename(bucket)
        try:
            f = open(filename, 'rb')
        except (FileNotFoundError, IsADirectoryError, PermissionError):
            return
        with f:
            bucket.load_bytecode(f)

    def dump_bytecode(self, bucket: Bucket) -> None:
        name = self._get_cache_filename(bucket)
        f = tempfile.NamedTemporaryFile(mode='wb', dir=os.path.dirname(name), prefix=os.path.basename(name), suffix='.tmp', delete=False)

        def remove_silent() -> None:
            try:
                os.remove(f.name)
            except OSError:
                pass
        try:
            with f:
                bucket.write_bytecode(f)
        except BaseException:
            remove_silent()
            raise
        try:
            os.replace(f.name, name)
        except OSError:
            remove_silent()
        except BaseException:
            remove_silent()
            raise

    def clear(self) -> None:
        from os import remove
        files = fnmatch.filter(os.listdir(self.directory), self.pattern % ('*',))
        for filename in files:
            try:
                remove(os.path.join(self.directory, filename))
            except OSError:
                pass