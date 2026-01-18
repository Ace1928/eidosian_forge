from builtins import open as bltn_open
import sys
import os
import io
import shutil
import stat
import time
import struct
import copy
import re
import warnings
class TarFile(object):
    """The TarFile Class provides an interface to tar archives.
    """
    debug = 0
    dereference = False
    ignore_zeros = False
    errorlevel = 1
    format = DEFAULT_FORMAT
    encoding = ENCODING
    errors = None
    tarinfo = TarInfo
    fileobject = ExFileObject
    extraction_filter = None

    def __init__(self, name=None, mode='r', fileobj=None, format=None, tarinfo=None, dereference=None, ignore_zeros=None, encoding=None, errors='surrogateescape', pax_headers=None, debug=None, errorlevel=None, copybufsize=None):
        """Open an (uncompressed) tar archive `name'. `mode' is either 'r' to
           read from an existing archive, 'a' to append data to an existing
           file or 'w' to create a new file overwriting an existing one. `mode'
           defaults to 'r'.
           If `fileobj' is given, it is used for reading or writing data. If it
           can be determined, `mode' is overridden by `fileobj's mode.
           `fileobj' is not closed, when TarFile is closed.
        """
        modes = {'r': 'rb', 'a': 'r+b', 'w': 'wb', 'x': 'xb'}
        if mode not in modes:
            raise ValueError("mode must be 'r', 'a', 'w' or 'x'")
        self.mode = mode
        self._mode = modes[mode]
        if not fileobj:
            if self.mode == 'a' and (not os.path.exists(name)):
                self.mode = 'w'
                self._mode = 'wb'
            fileobj = bltn_open(name, self._mode)
            self._extfileobj = False
        else:
            if name is None and hasattr(fileobj, 'name') and isinstance(fileobj.name, (str, bytes)):
                name = fileobj.name
            if hasattr(fileobj, 'mode'):
                self._mode = fileobj.mode
            self._extfileobj = True
        self.name = os.path.abspath(name) if name else None
        self.fileobj = fileobj
        if format is not None:
            self.format = format
        if tarinfo is not None:
            self.tarinfo = tarinfo
        if dereference is not None:
            self.dereference = dereference
        if ignore_zeros is not None:
            self.ignore_zeros = ignore_zeros
        if encoding is not None:
            self.encoding = encoding
        self.errors = errors
        if pax_headers is not None and self.format == PAX_FORMAT:
            self.pax_headers = pax_headers
        else:
            self.pax_headers = {}
        if debug is not None:
            self.debug = debug
        if errorlevel is not None:
            self.errorlevel = errorlevel
        self.copybufsize = copybufsize
        self.closed = False
        self.members = []
        self._loaded = False
        self.offset = self.fileobj.tell()
        self.inodes = {}
        try:
            if self.mode == 'r':
                self.firstmember = None
                self.firstmember = self.next()
            if self.mode == 'a':
                while True:
                    self.fileobj.seek(self.offset)
                    try:
                        tarinfo = self.tarinfo.fromtarfile(self)
                        self.members.append(tarinfo)
                    except EOFHeaderError:
                        self.fileobj.seek(self.offset)
                        break
                    except HeaderError as e:
                        raise ReadError(str(e)) from None
            if self.mode in ('a', 'w', 'x'):
                self._loaded = True
                if self.pax_headers:
                    buf = self.tarinfo.create_pax_global_header(self.pax_headers.copy())
                    self.fileobj.write(buf)
                    self.offset += len(buf)
        except:
            if not self._extfileobj:
                self.fileobj.close()
            self.closed = True
            raise

    @classmethod
    def open(cls, name=None, mode='r', fileobj=None, bufsize=RECORDSIZE, **kwargs):
        """Open a tar archive for reading, writing or appending. Return
           an appropriate TarFile class.

           mode:
           'r' or 'r:*' open for reading with transparent compression
           'r:'         open for reading exclusively uncompressed
           'r:gz'       open for reading with gzip compression
           'r:bz2'      open for reading with bzip2 compression
           'r:xz'       open for reading with lzma compression
           'a' or 'a:'  open for appending, creating the file if necessary
           'w' or 'w:'  open for writing without compression
           'w:gz'       open for writing with gzip compression
           'w:bz2'      open for writing with bzip2 compression
           'w:xz'       open for writing with lzma compression

           'x' or 'x:'  create a tarfile exclusively without compression, raise
                        an exception if the file is already created
           'x:gz'       create a gzip compressed tarfile, raise an exception
                        if the file is already created
           'x:bz2'      create a bzip2 compressed tarfile, raise an exception
                        if the file is already created
           'x:xz'       create an lzma compressed tarfile, raise an exception
                        if the file is already created

           'r|*'        open a stream of tar blocks with transparent compression
           'r|'         open an uncompressed stream of tar blocks for reading
           'r|gz'       open a gzip compressed stream of tar blocks
           'r|bz2'      open a bzip2 compressed stream of tar blocks
           'r|xz'       open an lzma compressed stream of tar blocks
           'w|'         open an uncompressed stream for writing
           'w|gz'       open a gzip compressed stream for writing
           'w|bz2'      open a bzip2 compressed stream for writing
           'w|xz'       open an lzma compressed stream for writing
        """
        if not name and (not fileobj):
            raise ValueError('nothing to open')
        if mode in ('r', 'r:*'):

            def not_compressed(comptype):
                return cls.OPEN_METH[comptype] == 'taropen'
            error_msgs = []
            for comptype in sorted(cls.OPEN_METH, key=not_compressed):
                func = getattr(cls, cls.OPEN_METH[comptype])
                if fileobj is not None:
                    saved_pos = fileobj.tell()
                try:
                    return func(name, 'r', fileobj, **kwargs)
                except (ReadError, CompressionError) as e:
                    error_msgs.append(f'- method {comptype}: {e!r}')
                    if fileobj is not None:
                        fileobj.seek(saved_pos)
                    continue
            error_msgs_summary = '\n'.join(error_msgs)
            raise ReadError(f'file could not be opened successfully:\n{error_msgs_summary}')
        elif ':' in mode:
            filemode, comptype = mode.split(':', 1)
            filemode = filemode or 'r'
            comptype = comptype or 'tar'
            if comptype in cls.OPEN_METH:
                func = getattr(cls, cls.OPEN_METH[comptype])
            else:
                raise CompressionError('unknown compression type %r' % comptype)
            return func(name, filemode, fileobj, **kwargs)
        elif '|' in mode:
            filemode, comptype = mode.split('|', 1)
            filemode = filemode or 'r'
            comptype = comptype or 'tar'
            if filemode not in ('r', 'w'):
                raise ValueError("mode must be 'r' or 'w'")
            stream = _Stream(name, filemode, comptype, fileobj, bufsize)
            try:
                t = cls(name, filemode, stream, **kwargs)
            except:
                stream.close()
                raise
            t._extfileobj = False
            return t
        elif mode in ('a', 'w', 'x'):
            return cls.taropen(name, mode, fileobj, **kwargs)
        raise ValueError('undiscernible mode')

    @classmethod
    def taropen(cls, name, mode='r', fileobj=None, **kwargs):
        """Open uncompressed tar archive name for reading or writing.
        """
        if mode not in ('r', 'a', 'w', 'x'):
            raise ValueError("mode must be 'r', 'a', 'w' or 'x'")
        return cls(name, mode, fileobj, **kwargs)

    @classmethod
    def gzopen(cls, name, mode='r', fileobj=None, compresslevel=9, **kwargs):
        """Open gzip compressed tar archive name for reading or writing.
           Appending is not allowed.
        """
        if mode not in ('r', 'w', 'x'):
            raise ValueError("mode must be 'r', 'w' or 'x'")
        try:
            from gzip import GzipFile
        except ImportError:
            raise CompressionError('gzip module is not available') from None
        try:
            fileobj = GzipFile(name, mode + 'b', compresslevel, fileobj)
        except OSError as e:
            if fileobj is not None and mode == 'r':
                raise ReadError('not a gzip file') from e
            raise
        try:
            t = cls.taropen(name, mode, fileobj, **kwargs)
        except OSError as e:
            fileobj.close()
            if mode == 'r':
                raise ReadError('not a gzip file') from e
            raise
        except:
            fileobj.close()
            raise
        t._extfileobj = False
        return t

    @classmethod
    def bz2open(cls, name, mode='r', fileobj=None, compresslevel=9, **kwargs):
        """Open bzip2 compressed tar archive name for reading or writing.
           Appending is not allowed.
        """
        if mode not in ('r', 'w', 'x'):
            raise ValueError("mode must be 'r', 'w' or 'x'")
        try:
            from bz2 import BZ2File
        except ImportError:
            raise CompressionError('bz2 module is not available') from None
        fileobj = BZ2File(fileobj or name, mode, compresslevel=compresslevel)
        try:
            t = cls.taropen(name, mode, fileobj, **kwargs)
        except (OSError, EOFError) as e:
            fileobj.close()
            if mode == 'r':
                raise ReadError('not a bzip2 file') from e
            raise
        except:
            fileobj.close()
            raise
        t._extfileobj = False
        return t

    @classmethod
    def xzopen(cls, name, mode='r', fileobj=None, preset=None, **kwargs):
        """Open lzma compressed tar archive name for reading or writing.
           Appending is not allowed.
        """
        if mode not in ('r', 'w', 'x'):
            raise ValueError("mode must be 'r', 'w' or 'x'")
        try:
            from lzma import LZMAFile, LZMAError
        except ImportError:
            raise CompressionError('lzma module is not available') from None
        fileobj = LZMAFile(fileobj or name, mode, preset=preset)
        try:
            t = cls.taropen(name, mode, fileobj, **kwargs)
        except (LZMAError, EOFError) as e:
            fileobj.close()
            if mode == 'r':
                raise ReadError('not an lzma file') from e
            raise
        except:
            fileobj.close()
            raise
        t._extfileobj = False
        return t
    OPEN_METH = {'tar': 'taropen', 'gz': 'gzopen', 'bz2': 'bz2open', 'xz': 'xzopen'}

    def close(self):
        """Close the TarFile. In write-mode, two finishing zero blocks are
           appended to the archive.
        """
        if self.closed:
            return
        self.closed = True
        try:
            if self.mode in ('a', 'w', 'x'):
                self.fileobj.write(NUL * (BLOCKSIZE * 2))
                self.offset += BLOCKSIZE * 2
                blocks, remainder = divmod(self.offset, RECORDSIZE)
                if remainder > 0:
                    self.fileobj.write(NUL * (RECORDSIZE - remainder))
        finally:
            if not self._extfileobj:
                self.fileobj.close()

    def getmember(self, name):
        """Return a TarInfo object for member `name'. If `name' can not be
           found in the archive, KeyError is raised. If a member occurs more
           than once in the archive, its last occurrence is assumed to be the
           most up-to-date version.
        """
        tarinfo = self._getmember(name.rstrip('/'))
        if tarinfo is None:
            raise KeyError('filename %r not found' % name)
        return tarinfo

    def getmembers(self):
        """Return the members of the archive as a list of TarInfo objects. The
           list has the same order as the members in the archive.
        """
        self._check()
        if not self._loaded:
            self._load()
        return self.members

    def getnames(self):
        """Return the members of the archive as a list of their names. It has
           the same order as the list returned by getmembers().
        """
        return [tarinfo.name for tarinfo in self.getmembers()]

    def gettarinfo(self, name=None, arcname=None, fileobj=None):
        """Create a TarInfo object from the result of os.stat or equivalent
           on an existing file. The file is either named by `name', or
           specified as a file object `fileobj' with a file descriptor. If
           given, `arcname' specifies an alternative name for the file in the
           archive, otherwise, the name is taken from the 'name' attribute of
           'fileobj', or the 'name' argument. The name should be a text
           string.
        """
        self._check('awx')
        if fileobj is not None:
            name = fileobj.name
        if arcname is None:
            arcname = name
        drv, arcname = os.path.splitdrive(arcname)
        arcname = arcname.replace(os.sep, '/')
        arcname = arcname.lstrip('/')
        tarinfo = self.tarinfo()
        tarinfo.tarfile = self
        if fileobj is None:
            if not self.dereference:
                statres = os.lstat(name)
            else:
                statres = os.stat(name)
        else:
            statres = os.fstat(fileobj.fileno())
        linkname = ''
        stmd = statres.st_mode
        if stat.S_ISREG(stmd):
            inode = (statres.st_ino, statres.st_dev)
            if not self.dereference and statres.st_nlink > 1 and (inode in self.inodes) and (arcname != self.inodes[inode]):
                type = LNKTYPE
                linkname = self.inodes[inode]
            else:
                type = REGTYPE
                if inode[0]:
                    self.inodes[inode] = arcname
        elif stat.S_ISDIR(stmd):
            type = DIRTYPE
        elif stat.S_ISFIFO(stmd):
            type = FIFOTYPE
        elif stat.S_ISLNK(stmd):
            type = SYMTYPE
            linkname = os.readlink(name)
        elif stat.S_ISCHR(stmd):
            type = CHRTYPE
        elif stat.S_ISBLK(stmd):
            type = BLKTYPE
        else:
            return None
        tarinfo.name = arcname
        tarinfo.mode = stmd
        tarinfo.uid = statres.st_uid
        tarinfo.gid = statres.st_gid
        if type == REGTYPE:
            tarinfo.size = statres.st_size
        else:
            tarinfo.size = 0
        tarinfo.mtime = statres.st_mtime
        tarinfo.type = type
        tarinfo.linkname = linkname
        if pwd:
            try:
                tarinfo.uname = pwd.getpwuid(tarinfo.uid)[0]
            except KeyError:
                pass
        if grp:
            try:
                tarinfo.gname = grp.getgrgid(tarinfo.gid)[0]
            except KeyError:
                pass
        if type in (CHRTYPE, BLKTYPE):
            if hasattr(os, 'major') and hasattr(os, 'minor'):
                tarinfo.devmajor = os.major(statres.st_rdev)
                tarinfo.devminor = os.minor(statres.st_rdev)
        return tarinfo

    def list(self, verbose=True, *, members=None):
        """Print a table of contents to sys.stdout. If `verbose' is False, only
           the names of the members are printed. If it is True, an `ls -l'-like
           output is produced. `members' is optional and must be a subset of the
           list returned by getmembers().
        """
        self._check()
        if members is None:
            members = self
        for tarinfo in members:
            if verbose:
                if tarinfo.mode is None:
                    _safe_print('??????????')
                else:
                    _safe_print(stat.filemode(tarinfo.mode))
                _safe_print('%s/%s' % (tarinfo.uname or tarinfo.uid, tarinfo.gname or tarinfo.gid))
                if tarinfo.ischr() or tarinfo.isblk():
                    _safe_print('%10s' % ('%d,%d' % (tarinfo.devmajor, tarinfo.devminor)))
                else:
                    _safe_print('%10d' % tarinfo.size)
                if tarinfo.mtime is None:
                    _safe_print('????-??-?? ??:??:??')
                else:
                    _safe_print('%d-%02d-%02d %02d:%02d:%02d' % time.localtime(tarinfo.mtime)[:6])
            _safe_print(tarinfo.name + ('/' if tarinfo.isdir() else ''))
            if verbose:
                if tarinfo.issym():
                    _safe_print('-> ' + tarinfo.linkname)
                if tarinfo.islnk():
                    _safe_print('link to ' + tarinfo.linkname)
            print()

    def add(self, name, arcname=None, recursive=True, *, filter=None):
        """Add the file `name' to the archive. `name' may be any type of file
           (directory, fifo, symbolic link, etc.). If given, `arcname'
           specifies an alternative name for the file in the archive.
           Directories are added recursively by default. This can be avoided by
           setting `recursive' to False. `filter' is a function
           that expects a TarInfo object argument and returns the changed
           TarInfo object, if it returns None the TarInfo object will be
           excluded from the archive.
        """
        self._check('awx')
        if arcname is None:
            arcname = name
        if self.name is not None and os.path.abspath(name) == self.name:
            self._dbg(2, 'tarfile: Skipped %r' % name)
            return
        self._dbg(1, name)
        tarinfo = self.gettarinfo(name, arcname)
        if tarinfo is None:
            self._dbg(1, 'tarfile: Unsupported type %r' % name)
            return
        if filter is not None:
            tarinfo = filter(tarinfo)
            if tarinfo is None:
                self._dbg(2, 'tarfile: Excluded %r' % name)
                return
        if tarinfo.isreg():
            with bltn_open(name, 'rb') as f:
                self.addfile(tarinfo, f)
        elif tarinfo.isdir():
            self.addfile(tarinfo)
            if recursive:
                for f in sorted(os.listdir(name)):
                    self.add(os.path.join(name, f), os.path.join(arcname, f), recursive, filter=filter)
        else:
            self.addfile(tarinfo)

    def addfile(self, tarinfo, fileobj=None):
        """Add the TarInfo object `tarinfo' to the archive. If `fileobj' is
           given, it should be a binary file, and tarinfo.size bytes are read
           from it and added to the archive. You can create TarInfo objects
           directly, or by using gettarinfo().
        """
        self._check('awx')
        tarinfo = copy.copy(tarinfo)
        buf = tarinfo.tobuf(self.format, self.encoding, self.errors)
        self.fileobj.write(buf)
        self.offset += len(buf)
        bufsize = self.copybufsize
        if fileobj is not None:
            copyfileobj(fileobj, self.fileobj, tarinfo.size, bufsize=bufsize)
            blocks, remainder = divmod(tarinfo.size, BLOCKSIZE)
            if remainder > 0:
                self.fileobj.write(NUL * (BLOCKSIZE - remainder))
                blocks += 1
            self.offset += blocks * BLOCKSIZE
        self.members.append(tarinfo)

    def _get_filter_function(self, filter):
        if filter is None:
            filter = self.extraction_filter
            if filter is None:
                return fully_trusted_filter
            if isinstance(filter, str):
                raise TypeError('String names are not supported for ' + 'TarFile.extraction_filter. Use a function such as ' + 'tarfile.data_filter directly.')
            return filter
        if callable(filter):
            return filter
        try:
            return _NAMED_FILTERS[filter]
        except KeyError:
            raise ValueError(f'filter {filter!r} not found') from None

    def extractall(self, path='.', members=None, *, numeric_owner=False, filter=None):
        """Extract all members from the archive to the current working
           directory and set owner, modification time and permissions on
           directories afterwards. `path' specifies a different directory
           to extract to. `members' is optional and must be a subset of the
           list returned by getmembers(). If `numeric_owner` is True, only
           the numbers for user/group names are used and not the names.

           The `filter` function will be called on each member just
           before extraction.
           It can return a changed TarInfo or None to skip the member.
           String names of common filters are accepted.
        """
        directories = []
        filter_function = self._get_filter_function(filter)
        if members is None:
            members = self
        for member in members:
            tarinfo = self._get_extract_tarinfo(member, filter_function, path)
            if tarinfo is None:
                continue
            if tarinfo.isdir():
                directories.append(tarinfo)
            self._extract_one(tarinfo, path, set_attrs=not tarinfo.isdir(), numeric_owner=numeric_owner)
        directories.sort(key=lambda a: a.name, reverse=True)
        for tarinfo in directories:
            dirpath = os.path.join(path, tarinfo.name)
            try:
                self.chown(tarinfo, dirpath, numeric_owner=numeric_owner)
                self.utime(tarinfo, dirpath)
                self.chmod(tarinfo, dirpath)
            except ExtractError as e:
                self._handle_nonfatal_error(e)

    def extract(self, member, path='', set_attrs=True, *, numeric_owner=False, filter=None):
        """Extract a member from the archive to the current working directory,
           using its full name. Its file information is extracted as accurately
           as possible. `member' may be a filename or a TarInfo object. You can
           specify a different directory using `path'. File attributes (owner,
           mtime, mode) are set unless `set_attrs' is False. If `numeric_owner`
           is True, only the numbers for user/group names are used and not
           the names.

           The `filter` function will be called before extraction.
           It can return a changed TarInfo or None to skip the member.
           String names of common filters are accepted.
        """
        filter_function = self._get_filter_function(filter)
        tarinfo = self._get_extract_tarinfo(member, filter_function, path)
        if tarinfo is not None:
            self._extract_one(tarinfo, path, set_attrs, numeric_owner)

    def _get_extract_tarinfo(self, member, filter_function, path):
        """Get filtered TarInfo (or None) from member, which might be a str"""
        if isinstance(member, str):
            tarinfo = self.getmember(member)
        else:
            tarinfo = member
        unfiltered = tarinfo
        try:
            tarinfo = filter_function(tarinfo, path)
        except (OSError, FilterError) as e:
            self._handle_fatal_error(e)
        except ExtractError as e:
            self._handle_nonfatal_error(e)
        if tarinfo is None:
            self._dbg(2, 'tarfile: Excluded %r' % unfiltered.name)
            return None
        if tarinfo.islnk():
            tarinfo = copy.copy(tarinfo)
            tarinfo._link_target = os.path.join(path, tarinfo.linkname)
        return tarinfo

    def _extract_one(self, tarinfo, path, set_attrs, numeric_owner):
        """Extract from filtered tarinfo to disk"""
        self._check('r')
        try:
            self._extract_member(tarinfo, os.path.join(path, tarinfo.name), set_attrs=set_attrs, numeric_owner=numeric_owner)
        except OSError as e:
            self._handle_fatal_error(e)
        except ExtractError as e:
            self._handle_nonfatal_error(e)

    def _handle_nonfatal_error(self, e):
        """Handle non-fatal error (ExtractError) according to errorlevel"""
        if self.errorlevel > 1:
            raise
        else:
            self._dbg(1, 'tarfile: %s' % e)

    def _handle_fatal_error(self, e):
        """Handle "fatal" error according to self.errorlevel"""
        if self.errorlevel > 0:
            raise
        elif isinstance(e, OSError):
            if e.filename is None:
                self._dbg(1, 'tarfile: %s' % e.strerror)
            else:
                self._dbg(1, 'tarfile: %s %r' % (e.strerror, e.filename))
        else:
            self._dbg(1, 'tarfile: %s %s' % (type(e).__name__, e))

    def extractfile(self, member):
        """Extract a member from the archive as a file object. `member' may be
           a filename or a TarInfo object. If `member' is a regular file or
           a link, an io.BufferedReader object is returned. For all other
           existing members, None is returned. If `member' does not appear
           in the archive, KeyError is raised.
        """
        self._check('r')
        if isinstance(member, str):
            tarinfo = self.getmember(member)
        else:
            tarinfo = member
        if tarinfo.isreg() or tarinfo.type not in SUPPORTED_TYPES:
            return self.fileobject(self, tarinfo)
        elif tarinfo.islnk() or tarinfo.issym():
            if isinstance(self.fileobj, _Stream):
                raise StreamError('cannot extract (sym)link as file object')
            else:
                return self.extractfile(self._find_link_target(tarinfo))
        else:
            return None

    def _extract_member(self, tarinfo, targetpath, set_attrs=True, numeric_owner=False):
        """Extract the TarInfo object tarinfo to a physical
           file called targetpath.
        """
        targetpath = targetpath.rstrip('/')
        targetpath = targetpath.replace('/', os.sep)
        upperdirs = os.path.dirname(targetpath)
        if upperdirs and (not os.path.exists(upperdirs)):
            os.makedirs(upperdirs)
        if tarinfo.islnk() or tarinfo.issym():
            self._dbg(1, '%s -> %s' % (tarinfo.name, tarinfo.linkname))
        else:
            self._dbg(1, tarinfo.name)
        if tarinfo.isreg():
            self.makefile(tarinfo, targetpath)
        elif tarinfo.isdir():
            self.makedir(tarinfo, targetpath)
        elif tarinfo.isfifo():
            self.makefifo(tarinfo, targetpath)
        elif tarinfo.ischr() or tarinfo.isblk():
            self.makedev(tarinfo, targetpath)
        elif tarinfo.islnk() or tarinfo.issym():
            self.makelink(tarinfo, targetpath)
        elif tarinfo.type not in SUPPORTED_TYPES:
            self.makeunknown(tarinfo, targetpath)
        else:
            self.makefile(tarinfo, targetpath)
        if set_attrs:
            self.chown(tarinfo, targetpath, numeric_owner)
            if not tarinfo.issym():
                self.chmod(tarinfo, targetpath)
                self.utime(tarinfo, targetpath)

    def makedir(self, tarinfo, targetpath):
        """Make a directory called targetpath.
        """
        try:
            if tarinfo.mode is None:
                os.mkdir(targetpath)
            else:
                os.mkdir(targetpath, 448)
        except FileExistsError:
            if not os.path.isdir(targetpath):
                raise

    def makefile(self, tarinfo, targetpath):
        """Make a file called targetpath.
        """
        source = self.fileobj
        source.seek(tarinfo.offset_data)
        bufsize = self.copybufsize
        with bltn_open(targetpath, 'wb') as target:
            if tarinfo.sparse is not None:
                for offset, size in tarinfo.sparse:
                    target.seek(offset)
                    copyfileobj(source, target, size, ReadError, bufsize)
                target.seek(tarinfo.size)
                target.truncate()
            else:
                copyfileobj(source, target, tarinfo.size, ReadError, bufsize)

    def makeunknown(self, tarinfo, targetpath):
        """Make a file from a TarInfo object with an unknown type
           at targetpath.
        """
        self.makefile(tarinfo, targetpath)
        self._dbg(1, 'tarfile: Unknown file type %r, extracted as regular file.' % tarinfo.type)

    def makefifo(self, tarinfo, targetpath):
        """Make a fifo called targetpath.
        """
        if hasattr(os, 'mkfifo'):
            os.mkfifo(targetpath)
        else:
            raise ExtractError('fifo not supported by system')

    def makedev(self, tarinfo, targetpath):
        """Make a character or block device called targetpath.
        """
        if not hasattr(os, 'mknod') or not hasattr(os, 'makedev'):
            raise ExtractError('special devices not supported by system')
        mode = tarinfo.mode
        if mode is None:
            mode = 384
        if tarinfo.isblk():
            mode |= stat.S_IFBLK
        else:
            mode |= stat.S_IFCHR
        os.mknod(targetpath, mode, os.makedev(tarinfo.devmajor, tarinfo.devminor))

    def makelink(self, tarinfo, targetpath):
        """Make a (symbolic) link called targetpath. If it cannot be created
          (platform limitation), we try to make a copy of the referenced file
          instead of a link.
        """
        try:
            if tarinfo.issym():
                if os.path.lexists(targetpath):
                    os.unlink(targetpath)
                os.symlink(tarinfo.linkname, targetpath)
            elif os.path.exists(tarinfo._link_target):
                os.link(tarinfo._link_target, targetpath)
            else:
                self._extract_member(self._find_link_target(tarinfo), targetpath)
        except symlink_exception:
            try:
                self._extract_member(self._find_link_target(tarinfo), targetpath)
            except KeyError:
                raise ExtractError('unable to resolve link inside archive') from None

    def chown(self, tarinfo, targetpath, numeric_owner):
        """Set owner of targetpath according to tarinfo. If numeric_owner
           is True, use .gid/.uid instead of .gname/.uname. If numeric_owner
           is False, fall back to .gid/.uid when the search based on name
           fails.
        """
        if hasattr(os, 'geteuid') and os.geteuid() == 0:
            g = tarinfo.gid
            u = tarinfo.uid
            if not numeric_owner:
                try:
                    if grp and tarinfo.gname:
                        g = grp.getgrnam(tarinfo.gname)[2]
                except KeyError:
                    pass
                try:
                    if pwd and tarinfo.uname:
                        u = pwd.getpwnam(tarinfo.uname)[2]
                except KeyError:
                    pass
            if g is None:
                g = -1
            if u is None:
                u = -1
            try:
                if tarinfo.issym() and hasattr(os, 'lchown'):
                    os.lchown(targetpath, u, g)
                else:
                    os.chown(targetpath, u, g)
            except OSError as e:
                raise ExtractError('could not change owner') from e

    def chmod(self, tarinfo, targetpath):
        """Set file permissions of targetpath according to tarinfo.
        """
        if tarinfo.mode is None:
            return
        try:
            os.chmod(targetpath, tarinfo.mode)
        except OSError as e:
            raise ExtractError('could not change mode') from e

    def utime(self, tarinfo, targetpath):
        """Set modification time of targetpath according to tarinfo.
        """
        mtime = tarinfo.mtime
        if mtime is None:
            return
        if not hasattr(os, 'utime'):
            return
        try:
            os.utime(targetpath, (mtime, mtime))
        except OSError as e:
            raise ExtractError('could not change modification time') from e

    def next(self):
        """Return the next member of the archive as a TarInfo object, when
           TarFile is opened for reading. Return None if there is no more
           available.
        """
        self._check('ra')
        if self.firstmember is not None:
            m = self.firstmember
            self.firstmember = None
            return m
        if self.offset != self.fileobj.tell():
            if self.offset == 0:
                return None
            self.fileobj.seek(self.offset - 1)
            if not self.fileobj.read(1):
                raise ReadError('unexpected end of data')
        tarinfo = None
        while True:
            try:
                tarinfo = self.tarinfo.fromtarfile(self)
            except EOFHeaderError as e:
                if self.ignore_zeros:
                    self._dbg(2, '0x%X: %s' % (self.offset, e))
                    self.offset += BLOCKSIZE
                    continue
            except InvalidHeaderError as e:
                if self.ignore_zeros:
                    self._dbg(2, '0x%X: %s' % (self.offset, e))
                    self.offset += BLOCKSIZE
                    continue
                elif self.offset == 0:
                    raise ReadError(str(e)) from None
            except EmptyHeaderError:
                if self.offset == 0:
                    raise ReadError('empty file') from None
            except TruncatedHeaderError as e:
                if self.offset == 0:
                    raise ReadError(str(e)) from None
            except SubsequentHeaderError as e:
                raise ReadError(str(e)) from None
            except Exception as e:
                try:
                    import zlib
                    if isinstance(e, zlib.error):
                        raise ReadError(f'zlib error: {e}') from None
                    else:
                        raise e
                except ImportError:
                    raise e
            break
        if tarinfo is not None:
            self.members.append(tarinfo)
        else:
            self._loaded = True
        return tarinfo

    def _getmember(self, name, tarinfo=None, normalize=False):
        """Find an archive member by name from bottom to top.
           If tarinfo is given, it is used as the starting point.
        """
        members = self.getmembers()
        skipping = False
        if tarinfo is not None:
            try:
                index = members.index(tarinfo)
            except ValueError:
                skipping = True
            else:
                members = members[:index]
        if normalize:
            name = os.path.normpath(name)
        for member in reversed(members):
            if skipping:
                if tarinfo.offset == member.offset:
                    skipping = False
                continue
            if normalize:
                member_name = os.path.normpath(member.name)
            else:
                member_name = member.name
            if name == member_name:
                return member
        if skipping:
            raise ValueError(tarinfo)

    def _load(self):
        """Read through the entire archive file and look for readable
           members.
        """
        while True:
            tarinfo = self.next()
            if tarinfo is None:
                break
        self._loaded = True

    def _check(self, mode=None):
        """Check if TarFile is still open, and if the operation's mode
           corresponds to TarFile's mode.
        """
        if self.closed:
            raise OSError('%s is closed' % self.__class__.__name__)
        if mode is not None and self.mode not in mode:
            raise OSError('bad operation for mode %r' % self.mode)

    def _find_link_target(self, tarinfo):
        """Find the target member of a symlink or hardlink member in the
           archive.
        """
        if tarinfo.issym():
            linkname = '/'.join(filter(None, (os.path.dirname(tarinfo.name), tarinfo.linkname)))
            limit = None
        else:
            linkname = tarinfo.linkname
            limit = tarinfo
        member = self._getmember(linkname, tarinfo=limit, normalize=True)
        if member is None:
            raise KeyError('linkname %r not found' % linkname)
        return member

    def __iter__(self):
        """Provide an iterator object.
        """
        if self._loaded:
            yield from self.members
            return
        index = 0
        if self.firstmember is not None:
            tarinfo = self.next()
            index += 1
            yield tarinfo
        while True:
            if index < len(self.members):
                tarinfo = self.members[index]
            elif not self._loaded:
                tarinfo = self.next()
                if not tarinfo:
                    self._loaded = True
                    return
            else:
                return
            index += 1
            yield tarinfo

    def _dbg(self, level, msg):
        """Write debugging output to sys.stderr.
        """
        if level <= self.debug:
            print(msg, file=sys.stderr)

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, type, value, traceback):
        if type is None:
            self.close()
        else:
            if not self._extfileobj:
                self.fileobj.close()
            self.closed = True