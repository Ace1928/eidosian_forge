import collections.abc
import contextlib
import datetime
import errno
import functools
import io
import os
import pathlib
import queue
import re
import stat
import sys
import time
from multiprocessing import Process
from threading import Thread
from typing import IO, Any, BinaryIO, Collection, Dict, List, Optional, Tuple, Type, Union
import multivolumefile
from py7zr.archiveinfo import Folder, Header, SignatureHeader
from py7zr.callbacks import ExtractCallback
from py7zr.compressor import SupportedMethods, get_methods_names
from py7zr.exceptions import (
from py7zr.helpers import (
from py7zr.properties import DEFAULT_FILTERS, FILTER_DEFLATE64, MAGIC_7Z, get_default_blocksize, get_memory_limit
class SevenZipFile(contextlib.AbstractContextManager):
    """The SevenZipFile Class provides an interface to 7z archives."""

    def __init__(self, file: Union[BinaryIO, str, pathlib.Path], mode: str='r', *, filters: Optional[List[Dict[str, int]]]=None, dereference=False, password: Optional[str]=None, header_encryption: bool=False, blocksize: Optional[int]=None, mp: bool=False) -> None:
        if mode not in ('r', 'w', 'x', 'a'):
            raise ValueError("ZipFile requires mode 'r', 'w', 'x', or 'a'")
        if mode == 'x':
            raise NotImplementedError
        self.fp: BinaryIO
        self.mp = mp
        self.password_protected = password is not None
        if blocksize:
            self._block_size = blocksize
        else:
            self._block_size = get_default_blocksize()
        if isinstance(file, str):
            self._filePassed: bool = False
            self.filename: str = file
            if mode == 'r':
                self.fp = open(file, 'rb')
            elif mode == 'w':
                self.fp = open(file, 'w+b')
            elif mode == 'x':
                self.fp = open(file, 'x+b')
            elif mode == 'a':
                self.fp = open(file, 'r+b')
            else:
                raise ValueError('File open error.')
            self.mode = mode
        elif isinstance(file, pathlib.Path):
            self._filePassed = False
            self.filename = str(file)
            if mode == 'r':
                self.fp = file.open(mode='rb')
            elif mode == 'w':
                self.fp = file.open(mode='w+b')
            elif mode == 'x':
                self.fp = file.open(mode='x+b')
            elif mode == 'a':
                self.fp = file.open(mode='r+b')
            else:
                raise ValueError('File open error.')
            self.mode = mode
        elif isinstance(file, multivolumefile.MultiVolume):
            self._filePassed = True
            self.fp = file
            self.filename = None
            self.mode = mode
        elif isinstance(file, io.IOBase):
            self._filePassed = True
            self.fp = file
            self.filename = getattr(file, 'name', None)
            self.mode = mode
        else:
            raise TypeError('invalid file: {}'.format(type(file)))
        self.encoded_header_mode = True
        self.header_encryption = header_encryption
        self._fileRefCnt = 1
        try:
            if mode == 'r':
                self._real_get_contents(password)
                self.fp.seek(self.afterheader)
                self.worker = Worker(self.files, self.afterheader, self.header, self.mp)
            elif mode == 'w':
                self._prepare_write(filters, password)
            elif mode == 'a':
                self._real_get_contents(password)
                self._prepare_append(filters, password)
            else:
                raise ValueError("Mode must be 'r', 'w', 'x', or 'a'")
        except Exception as e:
            self._fpclose()
            raise e
        self._dict: Dict[str, IO[Any]] = {}
        self.dereference = dereference
        self.reporterd: Optional[Thread] = None
        self.q: queue.Queue[Any] = queue.Queue()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _fpclose(self) -> None:
        assert self._fileRefCnt > 0
        self._fileRefCnt -= 1
        if not self._fileRefCnt and (not self._filePassed):
            self.fp.close()

    def _real_get_contents(self, password) -> None:
        if not self._check_7zfile(self.fp):
            raise Bad7zFile('not a 7z file')
        self.sig_header = SignatureHeader.retrieve(self.fp)
        self.afterheader: int = self.fp.tell()
        self.fp.seek(self.sig_header.nextheaderofs, os.SEEK_CUR)
        buffer = io.BytesIO(self.fp.read(self.sig_header.nextheadersize))
        if self.sig_header.nextheadercrc != calculate_crc32(buffer.getvalue()):
            raise Bad7zFile('invalid header data')
        header = Header.retrieve(self.fp, buffer, self.afterheader, password)
        if header is None:
            return
        header._initilized = True
        self.header = header
        header.size += 32 + self.sig_header.nextheadersize
        buffer.close()
        self.files = ArchiveFileList()
        if getattr(self.header, 'files_info', None) is None:
            return
        if hasattr(self.header, 'main_streams') and self.header.main_streams is not None:
            folders = self.header.main_streams.unpackinfo.folders
            for folder in folders:
                folder.password = password
            packinfo = self.header.main_streams.packinfo
            packsizes = packinfo.packsizes
            subinfo = self.header.main_streams.substreamsinfo
            if subinfo is not None and subinfo.unpacksizes is not None:
                unpacksizes = subinfo.unpacksizes
            else:
                unpacksizes = [x.unpacksizes[-1] for x in folders]
        else:
            subinfo = None
            folders = None
            packinfo = None
            packsizes = []
            unpacksizes = [0]
        pstat = self.ParseStatus()
        pstat.src_pos = self.afterheader
        file_in_solid = 0
        for file_id, file_info in enumerate(self.header.files_info.files):
            if not file_info['emptystream'] and folders is not None:
                folder = folders[pstat.folder]
                numinstreams = max([coder.get('numinstreams', 1) for coder in folder.coders])
                maxsize, compressed, uncompressed, packsize, solid = self._get_fileinfo_sizes(pstat, subinfo, packinfo, folder, packsizes, unpacksizes, file_in_solid, numinstreams)
                pstat.input += 1
                folder.solid = solid
                file_info['folder'] = folder
                file_info['maxsize'] = maxsize
                file_info['compressed'] = compressed
                file_info['uncompressed'] = uncompressed
                file_info['packsizes'] = packsize
                if subinfo.digestsdefined[pstat.outstreams]:
                    file_info['digest'] = subinfo.digests[pstat.outstreams]
                if folder is None:
                    pstat.src_pos += file_info['compressed']
                else:
                    if folder.solid:
                        file_in_solid += 1
                    pstat.outstreams += 1
                    if folder.files is None:
                        folder.files = ArchiveFileList(offset=file_id)
                    folder.files.append(file_info)
                    if pstat.input >= subinfo.num_unpackstreams_folders[pstat.folder]:
                        file_in_solid = 0
                        pstat.src_pos += sum(packinfo.packsizes[pstat.stream:pstat.stream + numinstreams])
                        pstat.folder += 1
                        pstat.stream += numinstreams
                        pstat.input = 0
            else:
                file_info['folder'] = None
                file_info['maxsize'] = 0
                file_info['compressed'] = 0
                file_info['uncompressed'] = 0
                file_info['packsizes'] = [0]
            if 'filename' not in file_info:
                try:
                    basefilename = self.filename
                except AttributeError:
                    file_info['filename'] = 'contents'
                else:
                    if basefilename is not None:
                        fn, ext = os.path.splitext(os.path.basename(basefilename))
                        file_info['filename'] = fn
                    else:
                        file_info['filename'] = 'contents'
            self.files.append(file_info)
        if not self.password_protected and self.header.main_streams is not None:
            self.password_protected = any([SupportedMethods.needs_password(folder.coders) for folder in self.header.main_streams.unpackinfo.folders])

    def _extract(self, path: Optional[Any]=None, targets: Optional[Collection[str]]=None, return_dict: bool=False, callback: Optional[ExtractCallback]=None) -> Optional[Dict[str, IO[Any]]]:
        if callback is None:
            pass
        elif isinstance(callback, ExtractCallback):
            self.reporterd = Thread(target=self.reporter, args=(callback,), daemon=True)
            self.reporterd.start()
        else:
            raise ValueError('Callback specified is not an instance of subclass of py7zr.callbacks.ExtractCallback class')
        target_files: List[Tuple[pathlib.Path, Dict[str, Any]]] = []
        target_dirs: List[pathlib.Path] = []
        if path is not None:
            if isinstance(path, str):
                path = pathlib.Path(path)
            try:
                if not path.exists():
                    path.mkdir(parents=True)
                else:
                    pass
            except OSError as e:
                if e.errno == errno.EEXIST and path.is_dir():
                    pass
                else:
                    raise e
        if targets is not None:
            targets = set(targets)
        fnames: Dict[str, int] = {}
        self.q.put(('pre', None, None))
        for f in self.files:
            if targets is not None and f.filename not in targets:
                self.worker.register_filelike(f.id, None)
                continue
            if f.filename not in fnames:
                outname = f.filename
                fnames[f.filename] = 0
            else:
                outname = f.filename + '_%d' % fnames[f.filename]
                fnames[f.filename] += 1
            if path is None or path.is_absolute():
                outfilename = get_sanitized_output_path(outname, path)
            else:
                outfilename = get_sanitized_output_path(outname, pathlib.Path(os.getcwd()).joinpath(path))
            if return_dict:
                if f.is_directory or f.is_socket:
                    pass
                else:
                    fname = outfilename.as_posix()
                    _buf = io.BytesIO()
                    self._dict[fname] = _buf
                    self.worker.register_filelike(f.id, MemIO(_buf))
            elif f.is_directory:
                if not outfilename.exists():
                    target_dirs.append(outfilename)
                    target_files.append((outfilename, f.file_properties()))
                else:
                    pass
            elif f.is_socket:
                pass
            elif f.is_symlink or f.is_junction:
                self.worker.register_filelike(f.id, outfilename)
            else:
                self.worker.register_filelike(f.id, outfilename)
                target_files.append((outfilename, f.file_properties()))
        for target_dir in sorted(target_dirs):
            try:
                target_dir.mkdir(parents=True)
            except FileExistsError:
                if target_dir.is_dir():
                    pass
                elif target_dir.is_file():
                    raise DecompressionError('Directory {} is existed as a normal file.'.format(str(target_dir)))
                else:
                    raise DecompressionError('Directory {} making fails on unknown condition.'.format(str(target_dir)))
        if callback is not None:
            self.worker.extract(self.fp, path, parallel=not self.password_protected and (not self._filePassed), q=self.q)
        else:
            self.worker.extract(self.fp, path, parallel=not self.password_protected and (not self._filePassed))
        self.q.put(('post', None, None))
        if return_dict:
            return self._dict
        for outfilename, properties in target_files:
            lastmodified = None
            try:
                lastmodified = ArchiveTimestamp(properties['lastwritetime']).totimestamp()
            except KeyError:
                pass
            if lastmodified is not None:
                os.utime(str(outfilename), times=(lastmodified, lastmodified))
            if os.name == 'posix':
                st_mode = properties['posix_mode']
                if st_mode is not None:
                    outfilename.chmod(st_mode)
                    continue
            if properties['readonly'] and (not properties['is_directory']):
                ro_mask = 511 ^ (stat.S_IWRITE | stat.S_IWGRP | stat.S_IWOTH)
                outfilename.chmod(outfilename.stat().st_mode & ro_mask)
        return None

    def _prepare_append(self, filters, password):
        if password is not None and filters is None:
            filters = DEFAULT_FILTERS.ENCRYPTED_ARCHIVE_FILTER
        elif filters is None:
            filters = DEFAULT_FILTERS.ARCHIVE_FILTER
        else:
            for f in filters:
                if f['id'] == FILTER_DEFLATE64:
                    raise UnsupportedCompressionMethodError(filters, 'Compression with deflate64 is not supported.')
        self.header.filters = filters
        self.header.password = password
        if self.header.main_streams is not None:
            pos = self.afterheader + self.header.main_streams.packinfo.packpositions[-1]
        else:
            pos = self.afterheader
        self.fp.seek(pos)
        self.worker = Worker(self.files, pos, self.header, self.mp)

    def _prepare_write(self, filters, password):
        if password is not None and filters is None:
            filters = DEFAULT_FILTERS.ENCRYPTED_ARCHIVE_FILTER
        elif filters is None:
            filters = DEFAULT_FILTERS.ARCHIVE_FILTER
        self.files = ArchiveFileList()
        self.sig_header = SignatureHeader()
        self.sig_header._write_skelton(self.fp)
        self.afterheader = self.fp.tell()
        self.header = Header.build_header(filters, password)
        self.fp.seek(self.afterheader)
        self.worker = Worker(self.files, self.afterheader, self.header, self.mp)

    def _write_flush(self):
        if self.header is not None:
            if self.header._initialized:
                folder = self.header.main_streams.unpackinfo.folders[-1]
                self.worker.flush_archive(self.fp, folder)
            self._write_header()

    def _write_header(self):
        """Write header and update signature header."""
        header_pos, header_len, header_crc = self.header.write(self.fp, self.afterheader, encoded=self.encoded_header_mode, encrypted=self.header_encryption)
        self.sig_header.nextheaderofs = header_pos - self.afterheader
        self.sig_header.calccrc(header_len, header_crc)
        self.sig_header.write(self.fp)

    def _writeall(self, path, arcname):
        try:
            if path.is_symlink() and (not self.dereference):
                self.write(path, arcname)
            elif path.is_file():
                self.write(path, arcname)
            elif path.is_dir():
                if not path.samefile('.'):
                    self.write(path, arcname)
                for nm in sorted(os.listdir(str(path))):
                    arc = os.path.join(arcname, nm) if arcname is not None else None
                    self._writeall(path.joinpath(nm), arc)
            else:
                return
        except OSError as ose:
            if self.dereference and ose.errno in [errno.ELOOP]:
                return
            elif self.dereference and sys.platform == 'win32' and (ose.errno in [errno.ENOENT]):
                return
            else:
                raise

    class ParseStatus:

        def __init__(self, src_pos=0):
            self.src_pos = src_pos
            self.folder = 0
            self.outstreams = 0
            self.input = 0
            self.stream = 0

    def _get_fileinfo_sizes(self, pstat, subinfo, packinfo, folder, packsizes, unpacksizes, file_in_solid, numinstreams):
        if pstat.input == 0:
            folder.solid = subinfo.num_unpackstreams_folders[pstat.folder] > 1
        maxsize = folder.solid and packinfo.packsizes[pstat.stream] or None
        uncompressed = unpacksizes[pstat.outstreams]
        if file_in_solid > 0:
            compressed = None
        elif pstat.stream < len(packsizes):
            compressed = packsizes[pstat.stream]
        else:
            compressed = uncompressed
        packsize = packsizes[pstat.stream:pstat.stream + numinstreams]
        return (maxsize, compressed, uncompressed, packsize, folder.solid)

    def set_encoded_header_mode(self, mode: bool) -> None:
        if mode:
            self.encoded_header_mode = True
        else:
            self.encoded_header_mode = False
            self.header_encryption = False

    def set_encrypted_header(self, mode: bool) -> None:
        if mode:
            self.encoded_header_mode = True
            self.header_encryption = True
        else:
            self.header_encryption = False

    @staticmethod
    def _check_7zfile(fp: Union[BinaryIO, io.BufferedReader, io.IOBase]) -> bool:
        result = MAGIC_7Z == fp.read(len(MAGIC_7Z))[:len(MAGIC_7Z)]
        fp.seek(-len(MAGIC_7Z), 1)
        return result

    def _get_method_names(self) -> List[str]:
        try:
            return get_methods_names([folder.coders for folder in self.header.main_streams.unpackinfo.folders])
        except KeyError:
            raise UnsupportedCompressionMethodError(self.header.main_streams.unpackinfo.folders, 'Unknown method')

    def _read_digest(self, pos: int, size: int) -> int:
        self.fp.seek(pos)
        remaining_size = size
        digest = 0
        while remaining_size > 0:
            block = min(self._block_size, remaining_size)
            digest = calculate_crc32(self.fp.read(block), digest)
            remaining_size -= block
        return digest

    def _is_solid(self):
        for f in self.header.main_streams.substreamsinfo.num_unpackstreams_folders:
            if f > 1:
                return True
        return False

    def _var_release(self):
        self._dict = {}
        self.worker.close()
        del self.worker
        del self.files
        del self.header
        del self.sig_header

    @staticmethod
    def _make_file_info(target: pathlib.Path, arcname: Optional[str]=None, dereference=False) -> Dict[str, Any]:
        f: Dict[str, Any] = {}
        f['origin'] = target
        if arcname is not None:
            f['filename'] = pathlib.Path(arcname).as_posix()
        else:
            f['filename'] = target.as_posix()
        if sys.platform == 'win32':
            fstat = target.lstat()
            if target.is_symlink():
                if dereference:
                    fstat = target.stat()
                    if stat.S_ISDIR(fstat.st_mode):
                        f['emptystream'] = True
                        f['attributes'] = fstat.st_file_attributes & FILE_ATTRIBUTE_WINDOWS_MASK
                    else:
                        f['emptystream'] = False
                        f['attributes'] = stat.FILE_ATTRIBUTE_ARCHIVE
                        f['uncompressed'] = fstat.st_size
                else:
                    f['emptystream'] = False
                    f['attributes'] = fstat.st_file_attributes & FILE_ATTRIBUTE_WINDOWS_MASK
            elif target.is_dir():
                f['emptystream'] = True
                f['attributes'] = fstat.st_file_attributes & FILE_ATTRIBUTE_WINDOWS_MASK
            elif target.is_file():
                f['emptystream'] = False
                f['attributes'] = stat.FILE_ATTRIBUTE_ARCHIVE
                f['uncompressed'] = fstat.st_size
        elif sys.platform == 'darwin' or sys.platform.startswith('linux') or sys.platform.startswith('freebsd') or sys.platform.startswith('netbsd') or sys.platform.startswith('sunos') or (sys.platform == 'aix'):
            fstat = target.lstat()
            if target.is_symlink():
                if dereference:
                    fstat = target.stat()
                    if stat.S_ISDIR(fstat.st_mode):
                        f['emptystream'] = True
                        f['attributes'] = getattr(stat, 'FILE_ATTRIBUTE_DIRECTORY')
                        f['attributes'] |= FILE_ATTRIBUTE_UNIX_EXTENSION | stat.S_IFDIR << 16
                        f['attributes'] |= stat.S_IMODE(fstat.st_mode) << 16
                    else:
                        f['emptystream'] = False
                        f['attributes'] = getattr(stat, 'FILE_ATTRIBUTE_ARCHIVE')
                        f['attributes'] |= FILE_ATTRIBUTE_UNIX_EXTENSION | stat.S_IMODE(fstat.st_mode) << 16
                else:
                    f['emptystream'] = False
                    f['attributes'] = getattr(stat, 'FILE_ATTRIBUTE_ARCHIVE') | getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT')
                    f['attributes'] |= FILE_ATTRIBUTE_UNIX_EXTENSION | stat.S_IFLNK << 16
                    f['attributes'] |= stat.S_IMODE(fstat.st_mode) << 16
            elif target.is_dir():
                f['emptystream'] = True
                f['attributes'] = getattr(stat, 'FILE_ATTRIBUTE_DIRECTORY')
                f['attributes'] |= FILE_ATTRIBUTE_UNIX_EXTENSION | stat.S_IFDIR << 16
                f['attributes'] |= stat.S_IMODE(fstat.st_mode) << 16
            elif target.is_file():
                f['emptystream'] = False
                f['uncompressed'] = fstat.st_size
                f['attributes'] = getattr(stat, 'FILE_ATTRIBUTE_ARCHIVE')
                f['attributes'] |= FILE_ATTRIBUTE_UNIX_EXTENSION | stat.S_IMODE(fstat.st_mode) << 16
        else:
            fstat = target.stat()
            if target.is_dir():
                f['emptystream'] = True
                f['attributes'] = stat.FILE_ATTRIBUTE_DIRECTORY
            elif target.is_file():
                f['emptystream'] = False
                f['uncompressed'] = fstat.st_size
                f['attributes'] = stat.FILE_ATTRIBUTE_ARCHIVE
        f['creationtime'] = ArchiveTimestamp.from_datetime(fstat.st_ctime)
        f['lastwritetime'] = ArchiveTimestamp.from_datetime(fstat.st_mtime)
        f['lastaccesstime'] = ArchiveTimestamp.from_datetime(fstat.st_atime)
        return f

    def _make_file_info_from_name(self, bio, size: int, arcname: str) -> Dict[str, Any]:
        f: Dict[str, Any] = {}
        f['origin'] = None
        f['data'] = bio
        f['filename'] = pathlib.Path(arcname).as_posix()
        f['uncompressed'] = size
        f['emptystream'] = size == 0
        f['attributes'] = getattr(stat, 'FILE_ATTRIBUTE_ARCHIVE')
        f['creationtime'] = ArchiveTimestamp.from_now()
        f['lastwritetime'] = ArchiveTimestamp.from_now()
        return f

    def _sanitize_archive_arcname(self, arcname):
        if isinstance(arcname, str):
            path = arcname
        else:
            path = str(arcname)
        if path.startswith(('/', os.sep)):
            path = path.lstrip('/' + os.sep)
        if re.match('^[a-zA-Z]:', path):
            path = path[2:]
            if path.startswith(('/', os.sep)):
                path = path.lstrip('/' + os.sep)
        if os.path.isabs(path) or re.match('^[a-zA-Z]:', path):
            raise AbsolutePathError(arcname)
        return path

    def getnames(self) -> List[str]:
        """Return the members of the archive as a list of their names. It has
        the same order as the list returned by getmembers().
        """
        return list(map(lambda x: x.filename, self.files))

    def archiveinfo(self) -> ArchiveInfo:
        total_uncompressed = functools.reduce(lambda x, y: x + y, [f.uncompressed for f in self.files])
        if isinstance(self.fp, multivolumefile.MultiVolume):
            fname = self.fp.name
            fstat = self.fp.stat()
        else:
            fname = self.filename
            assert fname is not None
            fstat = os.stat(fname)
        return ArchiveInfo(fname, fstat, self.header.size, self._get_method_names(), self._is_solid(), len(self.header.main_streams.unpackinfo.folders), total_uncompressed)

    def needs_password(self) -> bool:
        return self.password_protected

    def list(self) -> List[FileInfo]:
        """Returns contents information"""
        alist: List[FileInfo] = []
        lastmodified: Optional[datetime.datetime] = None
        for f in self.files:
            if f.lastwritetime is not None:
                lastmodified = filetime_to_dt(f.lastwritetime)
            alist.append(FileInfo(f.filename, f.compressed, f.uncompressed, f.archivable, f.is_directory, lastmodified, f.crc32))
        return alist

    def readall(self) -> Optional[Dict[str, IO[Any]]]:
        self._dict = {}
        return self._extract(path=None, return_dict=True)

    def extractall(self, path: Optional[Any]=None, callback: Optional[ExtractCallback]=None) -> None:
        """Extract all members from the archive to the current working
        directory and set owner, modification time and permissions on
        directories afterwards. ``path`` specifies a different directory
        to extract to.
        """
        self._extract(path=path, return_dict=False, callback=callback)

    def read(self, targets: Optional[Collection[str]]=None) -> Optional[Dict[str, IO[Any]]]:
        self._dict = {}
        return self._extract(path=None, targets=targets, return_dict=True)

    def extract(self, path: Optional[Any]=None, targets: Optional[Collection[str]]=None) -> None:
        self._extract(path, targets, return_dict=False)

    def reporter(self, callback: ExtractCallback):
        while True:
            try:
                item: Optional[Tuple[str, str, str]] = self.q.get(timeout=1)
            except queue.Empty:
                pass
            else:
                if item is None:
                    break
                elif item[0] == 's':
                    callback.report_start(item[1], item[2])
                elif item[0] == 'u':
                    callback.report_update(item[2])
                elif item[0] == 'e':
                    callback.report_end(item[1], item[2])
                elif item[0] == 'pre':
                    callback.report_start_preparation()
                elif item[0] == 'post':
                    callback.report_postprocess()
                elif item[0] == 'w':
                    callback.report_warning(item[1])
                else:
                    pass
                self.q.task_done()

    def writeall(self, path: Union[pathlib.Path, str], arcname: Optional[str]=None):
        """Write files in target path into archive."""
        if isinstance(path, str):
            path = pathlib.Path(path)
        if not path.exists():
            raise ValueError('specified path does not exist.')
        if path.is_dir() or path.is_file():
            self._writeall(path, arcname)
        else:
            raise ValueError('specified path is not a directory or a file')

    def write(self, file: Union[pathlib.Path, str], arcname: Optional[str]=None):
        """Write single target file into archive."""
        if not isinstance(file, str) and (not isinstance(file, pathlib.Path)):
            raise ValueError('Unsupported file type.')
        if arcname is None:
            arcname = self._sanitize_archive_arcname(file)
        else:
            arcname = self._sanitize_archive_arcname(arcname)
        if isinstance(file, str):
            path = pathlib.Path(file)
        else:
            path = file
        folder = self.header.initialize()
        file_info = self._make_file_info(path, arcname, self.dereference)
        self.header.files_info.files.append(file_info)
        self.header.files_info.emptyfiles.append(file_info['emptystream'])
        self.files.append(file_info)
        self.worker.archive(self.fp, self.files, folder, deref=self.dereference)

    def writed(self, targets: Dict[str, IO[Any]]) -> None:
        for target, input in targets.items():
            self.writef(input, target)

    def writef(self, bio: IO[Any], arcname: str):
        if not check_archive_path(arcname):
            raise ValueError(f'Specified path is bad: {arcname}')
        return self._writef(bio, arcname)

    def _writef(self, bio: IO[Any], arcname: str):
        if isinstance(bio, io.BytesIO):
            size = bio.getbuffer().nbytes
        elif isinstance(bio, io.TextIOBase):
            raise ValueError('Unsupported file object type: please open file with Binary mode.')
        elif isinstance(bio, io.BufferedIOBase):
            current = bio.tell()
            bio.seek(0, os.SEEK_END)
            last = bio.tell()
            bio.seek(current, os.SEEK_SET)
            size = last - current
        else:
            raise ValueError('Wrong argument passed for argument bio.')
        if size > 0:
            folder = self.header.initialize()
            file_info = self._make_file_info_from_name(bio, size, arcname)
            self.header.files_info.files.append(file_info)
            self.header.files_info.emptyfiles.append(file_info['emptystream'])
            self.files.append(file_info)
            self.worker.archive(self.fp, self.files, folder, deref=False)
        else:
            file_info = self._make_file_info_from_name(bio, size, arcname)
            self.header.files_info.files.append(file_info)
            self.header.files_info.emptyfiles.append(file_info['emptystream'])
            self.files.append(file_info)

    def writestr(self, data: Union[str, bytes, bytearray, memoryview], arcname: str):
        if not check_archive_path(arcname):
            raise ValueError(f'Specified path is bad: {arcname}')
        return self._writestr(data, arcname)

    def _writestr(self, data: Union[str, bytes, bytearray, memoryview], arcname: str):
        if not isinstance(arcname, str):
            raise ValueError('Unsupported arcname')
        if isinstance(data, str):
            self._writef(io.BytesIO(data.encode('UTF-8')), arcname)
        elif isinstance(data, bytes) or isinstance(data, bytearray) or isinstance(data, memoryview):
            self._writef(io.BytesIO(bytes(data)), arcname)
        else:
            raise ValueError('Unsupported data type.')

    def close(self):
        """Flush all the data into archive and close it.
        When close py7zr start reading target and writing actual archive file.
        """
        if 'w' in self.mode:
            self._write_flush()
        if 'a' in self.mode:
            self._write_flush()
        if 'r' in self.mode:
            if self.reporterd is not None:
                self.q.put_nowait(None)
                self.reporterd.join(1)
                if self.reporterd.is_alive():
                    raise InternalError('Progress report thread terminate error.')
                self.reporterd = None
        self._fpclose()
        self._var_release()

    def reset(self) -> None:
        """
        When read mode, it reset file pointer, decompress worker and decompressor
        """
        if self.mode == 'r':
            self.fp.seek(self.afterheader)
            self.worker = Worker(self.files, self.afterheader, self.header, self.mp)
            if self.header.main_streams is not None and self.header.main_streams.unpackinfo.numfolders > 0:
                for i, folder in enumerate(self.header.main_streams.unpackinfo.folders):
                    folder.decompressor = None

    def test(self) -> Optional[bool]:
        self.fp.seek(self.afterheader)
        self.worker = Worker(self.files, self.afterheader, self.header, self.mp)
        crcs: Optional[List[int]] = self.header.main_streams.packinfo.crcs
        if crcs is None or len(crcs) == 0:
            return None
        packpos = self.afterheader + self.header.main_streams.packinfo.packpos
        packsizes = self.header.main_streams.packinfo.packsizes
        digestdefined = self.header.main_streams.packinfo.digestdefined
        j = 0
        for i, d in enumerate(digestdefined):
            if d:
                if self._read_digest(packpos, packsizes[i]) != crcs[j]:
                    return False
                j += 1
            packpos += packsizes[i]
        return True

    def testzip(self) -> Optional[str]:
        self.fp.seek(self.afterheader)
        self.worker = Worker(self.files, self.afterheader, self.header, self.mp)
        for f in self.files:
            self.worker.register_filelike(f.id, None)
        try:
            self.worker.extract(self.fp, None, parallel=not self.password_protected, skip_notarget=False)
        except CrcError as crce:
            return crce.args[2]
        else:
            return None