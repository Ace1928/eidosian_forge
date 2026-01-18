from __future__ import unicode_literals
import typing
import six
from . import errors
from .base import FS
from .copy import copy_dir, copy_file
from .error_tools import unwrap_errors
from .info import Info
from .path import abspath, join, normpath
@six.python_2_unicode_compatible
class WrapFS(FS, typing.Generic[_F]):
    """A proxy for a filesystem object.

    This class exposes an filesystem interface, where the data is
    stored on another filesystem(s), and is the basis for
    `~fs.subfs.SubFS` and other *virtual* filesystems.

    """
    wrap_name = None

    def __init__(self, wrap_fs):
        self._wrap_fs = wrap_fs
        super(WrapFS, self).__init__()

    def __repr__(self):
        return '{}({!r})'.format(self.__class__.__name__, self._wrap_fs)

    def __str__(self):
        wraps = []
        _fs = self
        while hasattr(_fs, '_wrap_fs'):
            wrap_name = getattr(_fs, 'wrap_name', None)
            if wrap_name is not None:
                wraps.append(wrap_name)
            _fs = _fs._wrap_fs
        if wraps:
            _str = '{}({})'.format(_fs, ', '.join(wraps[::-1]))
        else:
            _str = '{}'.format(_fs)
        return _str

    def delegate_path(self, path):
        """Encode a path for proxied filesystem.

        Arguments:
            path (str): A path on the filesystem.

        Returns:
            (FS, str): a tuple of ``(<filesystem>, <new_path>)``

        """
        return (self._wrap_fs, path)

    def delegate_fs(self):
        """Get the proxied filesystem.

        This method should return a filesystem for methods not
        associated with a path, e.g. `~fs.base.FS.getmeta`.

        """
        return self._wrap_fs

    def appendbytes(self, path, data):
        self.check()
        _fs, _path = self.delegate_path(path)
        with unwrap_errors(path):
            return _fs.appendbytes(_path, data)

    def appendtext(self, path, text, encoding='utf-8', errors=None, newline=''):
        self.check()
        _fs, _path = self.delegate_path(path)
        with unwrap_errors(path):
            return _fs.appendtext(_path, text, encoding=encoding, errors=errors, newline=newline)

    def getinfo(self, path, namespaces=None):
        self.check()
        _fs, _path = self.delegate_path(path)
        with unwrap_errors(path):
            raw_info = _fs.getinfo(_path, namespaces=namespaces).raw
        if abspath(normpath(path)) == '/':
            raw_info = dict(raw_info)
            raw_info['basic']['name'] = ''
        return Info(raw_info)

    def listdir(self, path):
        self.check()
        _fs, _path = self.delegate_path(path)
        with unwrap_errors(path):
            dir_list = _fs.listdir(_path)
        return dir_list

    def lock(self):
        self.check()
        _fs = self.delegate_fs()
        return _fs.lock()

    def makedir(self, path, permissions=None, recreate=False):
        self.check()
        _fs, _path = self.delegate_path(path)
        with unwrap_errors(path):
            return _fs.makedir(_path, permissions=permissions, recreate=recreate)

    def move(self, src_path, dst_path, overwrite=False, preserve_time=False):
        _fs, _src_path = self.delegate_path(src_path)
        _, _dst_path = self.delegate_path(dst_path)
        with unwrap_errors({_src_path: src_path, _dst_path: dst_path}):
            _fs.move(_src_path, _dst_path, overwrite=overwrite, preserve_time=preserve_time)

    def movedir(self, src_path, dst_path, create=False, preserve_time=False):
        _fs, _src_path = self.delegate_path(src_path)
        _, _dst_path = self.delegate_path(dst_path)
        with unwrap_errors({_src_path: src_path, _dst_path: dst_path}):
            _fs.movedir(_src_path, _dst_path, create=create, preserve_time=preserve_time)

    def openbin(self, path, mode='r', buffering=-1, **options):
        self.check()
        _fs, _path = self.delegate_path(path)
        with unwrap_errors(path):
            bin_file = _fs.openbin(_path, mode=mode, buffering=-1, **options)
        return bin_file

    def remove(self, path):
        self.check()
        _fs, _path = self.delegate_path(path)
        with unwrap_errors(path):
            _fs.remove(_path)

    def removedir(self, path):
        self.check()
        _path = abspath(normpath(path))
        if _path == '/':
            raise errors.RemoveRootError()
        _fs, _path = self.delegate_path(path)
        with unwrap_errors(path):
            _fs.removedir(_path)

    def removetree(self, dir_path):
        self.check()
        _path = abspath(normpath(dir_path))
        _delegate_fs, _delegate_path = self.delegate_path(dir_path)
        with unwrap_errors(dir_path):
            if _path == '/':
                for info in _delegate_fs.scandir(_delegate_path):
                    info_path = join(_delegate_path, info.name)
                    if info.is_dir:
                        _delegate_fs.removetree(info_path)
                    else:
                        _delegate_fs.remove(info_path)
            else:
                _delegate_fs.removetree(_delegate_path)

    def scandir(self, path, namespaces=None, page=None):
        self.check()
        _fs, _path = self.delegate_path(path)
        with unwrap_errors(path):
            for info in _fs.scandir(_path, namespaces=namespaces, page=page):
                yield info

    def setinfo(self, path, info):
        self.check()
        _fs, _path = self.delegate_path(path)
        return _fs.setinfo(_path, info)

    def settimes(self, path, accessed=None, modified=None):
        self.check()
        _fs, _path = self.delegate_path(path)
        with unwrap_errors(path):
            _fs.settimes(_path, accessed=accessed, modified=modified)

    def touch(self, path):
        self.check()
        _fs, _path = self.delegate_path(path)
        with unwrap_errors(path):
            _fs.touch(_path)

    def copy(self, src_path, dst_path, overwrite=False, preserve_time=False):
        src_fs, _src_path = self.delegate_path(src_path)
        dst_fs, _dst_path = self.delegate_path(dst_path)
        with unwrap_errors({_src_path: src_path, _dst_path: dst_path}):
            if not overwrite and dst_fs.exists(_dst_path):
                raise errors.DestinationExists(_dst_path)
            copy_file(src_fs, _src_path, dst_fs, _dst_path, preserve_time=preserve_time)

    def copydir(self, src_path, dst_path, create=False, preserve_time=False):
        src_fs, _src_path = self.delegate_path(src_path)
        dst_fs, _dst_path = self.delegate_path(dst_path)
        with unwrap_errors({_src_path: src_path, _dst_path: dst_path}):
            if not create and (not dst_fs.exists(_dst_path)):
                raise errors.ResourceNotFound(dst_path)
            if not src_fs.getinfo(_src_path).is_dir:
                raise errors.DirectoryExpected(src_path)
            copy_dir(src_fs, _src_path, dst_fs, _dst_path, preserve_time=preserve_time)

    def create(self, path, wipe=False):
        self.check()
        _fs, _path = self.delegate_path(path)
        with unwrap_errors(path):
            return _fs.create(_path, wipe=wipe)

    def desc(self, path):
        self.check()
        _fs, _path = self.delegate_path(path)
        with unwrap_errors(path):
            desc = _fs.desc(_path)
        return desc

    def download(self, path, file, chunk_size=None, **options):
        self.check()
        _fs, _path = self.delegate_path(path)
        with unwrap_errors(path):
            _fs.download(_path, file, chunk_size=chunk_size, **options)

    def exists(self, path):
        self.check()
        _fs, _path = self.delegate_path(path)
        with unwrap_errors(path):
            exists = _fs.exists(_path)
        return exists

    def filterdir(self, path, files=None, dirs=None, exclude_dirs=None, exclude_files=None, namespaces=None, page=None):
        self.check()
        _fs, _path = self.delegate_path(path)
        iter_files = iter(_fs.filterdir(_path, exclude_dirs=exclude_dirs, exclude_files=exclude_files, files=files, dirs=dirs, namespaces=namespaces, page=page))
        with unwrap_errors(path):
            for info in iter_files:
                yield info

    def readbytes(self, path):
        self.check()
        _fs, _path = self.delegate_path(path)
        with unwrap_errors(path):
            _bytes = _fs.readbytes(_path)
        return _bytes

    def readtext(self, path, encoding=None, errors=None, newline=''):
        self.check()
        _fs, _path = self.delegate_path(path)
        with unwrap_errors(path):
            _text = _fs.readtext(_path, encoding=encoding, errors=errors, newline=newline)
        return _text

    def getmeta(self, namespace='standard'):
        self.check()
        meta = self.delegate_fs().getmeta(namespace=namespace)
        return meta

    def getsize(self, path):
        self.check()
        _fs, _path = self.delegate_path(path)
        with unwrap_errors(path):
            size = _fs.getsize(_path)
        return size

    def getsyspath(self, path):
        self.check()
        _fs, _path = self.delegate_path(path)
        with unwrap_errors(path):
            sys_path = _fs.getsyspath(_path)
        return sys_path

    def gettype(self, path):
        self.check()
        _fs, _path = self.delegate_path(path)
        with unwrap_errors(path):
            _type = _fs.gettype(_path)
        return _type

    def geturl(self, path, purpose='download'):
        self.check()
        _fs, _path = self.delegate_path(path)
        with unwrap_errors(path):
            return _fs.geturl(_path, purpose=purpose)

    def hassyspath(self, path):
        self.check()
        _fs, _path = self.delegate_path(path)
        with unwrap_errors(path):
            has_sys_path = _fs.hassyspath(_path)
        return has_sys_path

    def hasurl(self, path, purpose='download'):
        self.check()
        _fs, _path = self.delegate_path(path)
        with unwrap_errors(path):
            has_url = _fs.hasurl(_path, purpose=purpose)
        return has_url

    def isdir(self, path):
        self.check()
        _fs, _path = self.delegate_path(path)
        with unwrap_errors(path):
            _isdir = _fs.isdir(_path)
        return _isdir

    def isfile(self, path):
        self.check()
        _fs, _path = self.delegate_path(path)
        with unwrap_errors(path):
            _isfile = _fs.isfile(_path)
        return _isfile

    def islink(self, path):
        self.check()
        _fs, _path = self.delegate_path(path)
        with unwrap_errors(path):
            _islink = _fs.islink(_path)
        return _islink

    def makedirs(self, path, permissions=None, recreate=False):
        self.check()
        _fs, _path = self.delegate_path(path)
        return _fs.makedirs(_path, permissions=permissions, recreate=recreate)

    def open(self, path, mode='r', buffering=-1, encoding=None, errors=None, newline='', line_buffering=False, **options):
        self.check()
        _fs, _path = self.delegate_path(path)
        with unwrap_errors(path):
            open_file = _fs.open(_path, mode=mode, buffering=buffering, encoding=encoding, errors=errors, newline=newline, line_buffering=line_buffering, **options)
        return open_file

    def opendir(self, path, factory=None):
        from .subfs import SubFS
        factory = factory or SubFS
        if not self.getinfo(path).is_dir:
            raise errors.DirectoryExpected(path=path)
        with unwrap_errors(path):
            return factory(self, path)

    def writebytes(self, path, contents):
        self.check()
        _fs, _path = self.delegate_path(path)
        with unwrap_errors(path):
            _fs.writebytes(_path, contents)

    def upload(self, path, file, chunk_size=None, **options):
        self.check()
        _fs, _path = self.delegate_path(path)
        with unwrap_errors(path):
            _fs.upload(_path, file, chunk_size=chunk_size, **options)

    def writefile(self, path, file, encoding=None, errors=None, newline=''):
        self.check()
        _fs, _path = self.delegate_path(path)
        with unwrap_errors(path):
            _fs.writefile(_path, file, encoding=encoding, errors=errors, newline=newline)

    def validatepath(self, path):
        self.check()
        _fs, _path = self.delegate_path(path)
        with unwrap_errors(path):
            _fs.validatepath(_path)
        path = abspath(normpath(path))
        return path

    def hash(self, path, name):
        self.check()
        _fs, _path = self.delegate_path(path)
        with unwrap_errors(path):
            return _fs.hash(_path, name)

    @property
    def walk(self):
        return self._wrap_fs.walker_class.bind(self)