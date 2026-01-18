from contextlib import contextmanager
from ctypes import (
import libarchive
import libarchive.ffi as ffi
from fsspec import open_files
from fsspec.archive import AbstractArchiveFileSystem
from fsspec.implementations.memory import MemoryFile
from fsspec.utils import DEFAULT_BLOCK_SIZE
class LibArchiveFileSystem(AbstractArchiveFileSystem):
    """Compressed archives as a file-system (read-only)

    Supports the following formats:
    tar, pax , cpio, ISO9660, zip, mtree, shar, ar, raw, xar, lha/lzh, rar
    Microsoft CAB, 7-Zip, WARC

    See the libarchive documentation for further restrictions.
    https://www.libarchive.org/

    Keeps file object open while instance lives. It only works in seekable
    file-like objects. In case the filesystem does not support this kind of
    file object, it is recommended to cache locally.

    This class is pickleable, but not necessarily thread-safe (depends on the
    platform). See libarchive documentation for details.
    """
    root_marker = ''
    protocol = 'libarchive'
    cachable = False

    def __init__(self, fo='', mode='r', target_protocol=None, target_options=None, block_size=DEFAULT_BLOCK_SIZE, **kwargs):
        """
        Parameters
        ----------
        fo: str or file-like
            Contains ZIP, and must exist. If a str, will fetch file using
            :meth:`~fsspec.open_files`, which must return one file exactly.
        mode: str
            Currently, only 'r' accepted
        target_protocol: str (optional)
            If ``fo`` is a string, this value can be used to override the
            FS protocol inferred from a URL
        target_options: dict (optional)
            Kwargs passed when instantiating the target FS, if ``fo`` is
            a string.
        """
        super().__init__(self, **kwargs)
        if mode != 'r':
            raise ValueError('Only read from archive files accepted')
        if isinstance(fo, str):
            files = open_files(fo, protocol=target_protocol, **target_options or {})
            if len(files) != 1:
                raise ValueError(f'Path "{fo}" did not resolve to exactly one file: "{files}"')
            fo = files[0]
        self.of = fo
        self.fo = fo.__enter__()
        self.block_size = block_size
        self.dir_cache = None

    @contextmanager
    def _open_archive(self):
        self.fo.seek(0)
        with custom_reader(self.fo, block_size=self.block_size) as arc:
            yield arc

    @classmethod
    def _strip_protocol(cls, path):
        return super()._strip_protocol(path).lstrip('/')

    def _get_dirs(self):
        fields = {'name': 'pathname', 'size': 'size', 'created': 'ctime', 'mode': 'mode', 'uid': 'uid', 'gid': 'gid', 'mtime': 'mtime'}
        if self.dir_cache is not None:
            return
        self.dir_cache = {}
        list_names = []
        with self._open_archive() as arc:
            for entry in arc:
                if not entry.isdir and (not entry.isfile):
                    continue
                self.dir_cache.update({dirname: {'name': dirname, 'size': 0, 'type': 'directory'} for dirname in self._all_dirnames(set(entry.name))})
                f = {key: getattr(entry, fields[key]) for key in fields}
                f['type'] = 'directory' if entry.isdir else 'file'
                list_names.append(entry.name)
                self.dir_cache[f['name']] = f
        self.dir_cache.update({dirname: {'name': dirname, 'size': 0, 'type': 'directory'} for dirname in self._all_dirnames(list_names)})

    def _open(self, path, mode='rb', block_size=None, autocommit=True, cache_options=None, **kwargs):
        path = self._strip_protocol(path)
        if mode != 'rb':
            raise NotImplementedError
        data = bytes()
        with self._open_archive() as arc:
            for entry in arc:
                if entry.pathname != path:
                    continue
                if entry.size == 0:
                    break
                for block in entry.get_blocks(entry.size):
                    data = block
                    break
                else:
                    raise ValueError
        return MemoryFile(fs=self, path=path, data=data)