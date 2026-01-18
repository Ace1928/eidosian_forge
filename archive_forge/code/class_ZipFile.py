import binascii
import importlib.util
import io
import itertools
import os
import posixpath
import shutil
import stat
import struct
import sys
import threading
import time
import contextlib
import pathlib
class ZipFile:
    """ Class with methods to open, read, write, close, list zip files.

    z = ZipFile(file, mode="r", compression=ZIP_STORED, allowZip64=True,
                compresslevel=None)

    file: Either the path to the file, or a file-like object.
          If it is a path, the file will be opened and closed by ZipFile.
    mode: The mode can be either read 'r', write 'w', exclusive create 'x',
          or append 'a'.
    compression: ZIP_STORED (no compression), ZIP_DEFLATED (requires zlib),
                 ZIP_BZIP2 (requires bz2) or ZIP_LZMA (requires lzma).
    allowZip64: if True ZipFile will create files with ZIP64 extensions when
                needed, otherwise it will raise an exception when this would
                be necessary.
    compresslevel: None (default for the given compression type) or an integer
                   specifying the level to pass to the compressor.
                   When using ZIP_STORED or ZIP_LZMA this keyword has no effect.
                   When using ZIP_DEFLATED integers 0 through 9 are accepted.
                   When using ZIP_BZIP2 integers 1 through 9 are accepted.

    """
    fp = None
    _windows_illegal_name_trans_table = None

    def __init__(self, file, mode='r', compression=ZIP_STORED, allowZip64=True, compresslevel=None, *, strict_timestamps=True, metadata_encoding=None):
        """Open the ZIP file with mode read 'r', write 'w', exclusive create 'x',
        or append 'a'."""
        if mode not in ('r', 'w', 'x', 'a'):
            raise ValueError("ZipFile requires mode 'r', 'w', 'x', or 'a'")
        _check_compression(compression)
        self._allowZip64 = allowZip64
        self._didModify = False
        self.debug = 0
        self.NameToInfo = {}
        self.filelist = []
        self.compression = compression
        self.compresslevel = compresslevel
        self.mode = mode
        self.pwd = None
        self._comment = b''
        self._strict_timestamps = strict_timestamps
        self.metadata_encoding = metadata_encoding
        if self.metadata_encoding and mode != 'r':
            raise ValueError('metadata_encoding is only supported for reading files')
        if isinstance(file, os.PathLike):
            file = os.fspath(file)
        if isinstance(file, str):
            self._filePassed = 0
            self.filename = file
            modeDict = {'r': 'rb', 'w': 'w+b', 'x': 'x+b', 'a': 'r+b', 'r+b': 'w+b', 'w+b': 'wb', 'x+b': 'xb'}
            filemode = modeDict[mode]
            while True:
                try:
                    self.fp = io.open(file, filemode)
                except OSError:
                    if filemode in modeDict:
                        filemode = modeDict[filemode]
                        continue
                    raise
                break
        else:
            self._filePassed = 1
            self.fp = file
            self.filename = getattr(file, 'name', None)
        self._fileRefCnt = 1
        self._lock = threading.RLock()
        self._seekable = True
        self._writing = False
        try:
            if mode == 'r':
                self._RealGetContents()
            elif mode in ('w', 'x'):
                self._didModify = True
                try:
                    self.start_dir = self.fp.tell()
                except (AttributeError, OSError):
                    self.fp = _Tellable(self.fp)
                    self.start_dir = 0
                    self._seekable = False
                else:
                    try:
                        self.fp.seek(self.start_dir)
                    except (AttributeError, OSError):
                        self._seekable = False
            elif mode == 'a':
                try:
                    self._RealGetContents()
                    self.fp.seek(self.start_dir)
                except BadZipFile:
                    self.fp.seek(0, 2)
                    self._didModify = True
                    self.start_dir = self.fp.tell()
            else:
                raise ValueError("Mode must be 'r', 'w', 'x', or 'a'")
        except:
            fp = self.fp
            self.fp = None
            self._fpclose(fp)
            raise

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __repr__(self):
        result = ['<%s.%s' % (self.__class__.__module__, self.__class__.__qualname__)]
        if self.fp is not None:
            if self._filePassed:
                result.append(' file=%r' % self.fp)
            elif self.filename is not None:
                result.append(' filename=%r' % self.filename)
            result.append(' mode=%r' % self.mode)
        else:
            result.append(' [closed]')
        result.append('>')
        return ''.join(result)

    def _RealGetContents(self):
        """Read in the table of contents for the ZIP file."""
        fp = self.fp
        try:
            endrec = _EndRecData(fp)
        except OSError:
            raise BadZipFile('File is not a zip file')
        if not endrec:
            raise BadZipFile('File is not a zip file')
        if self.debug > 1:
            print(endrec)
        size_cd = endrec[_ECD_SIZE]
        offset_cd = endrec[_ECD_OFFSET]
        self._comment = endrec[_ECD_COMMENT]
        concat = endrec[_ECD_LOCATION] - size_cd - offset_cd
        if endrec[_ECD_SIGNATURE] == stringEndArchive64:
            concat -= sizeEndCentDir64 + sizeEndCentDir64Locator
        if self.debug > 2:
            inferred = concat + offset_cd
            print('given, inferred, offset', offset_cd, inferred, concat)
        self.start_dir = offset_cd + concat
        if self.start_dir < 0:
            raise BadZipFile('Bad offset for central directory')
        fp.seek(self.start_dir, 0)
        data = fp.read(size_cd)
        fp = io.BytesIO(data)
        total = 0
        while total < size_cd:
            centdir = fp.read(sizeCentralDir)
            if len(centdir) != sizeCentralDir:
                raise BadZipFile('Truncated central directory')
            centdir = struct.unpack(structCentralDir, centdir)
            if centdir[_CD_SIGNATURE] != stringCentralDir:
                raise BadZipFile('Bad magic number for central directory')
            if self.debug > 2:
                print(centdir)
            filename = fp.read(centdir[_CD_FILENAME_LENGTH])
            flags = centdir[_CD_FLAG_BITS]
            if flags & _MASK_UTF_FILENAME:
                filename = filename.decode('utf-8')
            else:
                filename = filename.decode(self.metadata_encoding or 'cp437')
            x = ZipInfo(filename)
            x.extra = fp.read(centdir[_CD_EXTRA_FIELD_LENGTH])
            x.comment = fp.read(centdir[_CD_COMMENT_LENGTH])
            x.header_offset = centdir[_CD_LOCAL_HEADER_OFFSET]
            x.create_version, x.create_system, x.extract_version, x.reserved, x.flag_bits, x.compress_type, t, d, x.CRC, x.compress_size, x.file_size = centdir[1:12]
            if x.extract_version > MAX_EXTRACT_VERSION:
                raise NotImplementedError('zip file version %.1f' % (x.extract_version / 10))
            x.volume, x.internal_attr, x.external_attr = centdir[15:18]
            x._raw_time = t
            x.date_time = ((d >> 9) + 1980, d >> 5 & 15, d & 31, t >> 11, t >> 5 & 63, (t & 31) * 2)
            x._decodeExtra()
            x.header_offset = x.header_offset + concat
            self.filelist.append(x)
            self.NameToInfo[x.filename] = x
            total = total + sizeCentralDir + centdir[_CD_FILENAME_LENGTH] + centdir[_CD_EXTRA_FIELD_LENGTH] + centdir[_CD_COMMENT_LENGTH]
            if self.debug > 2:
                print('total', total)
        end_offset = self.start_dir
        for zinfo in sorted(self.filelist, key=lambda zinfo: zinfo.header_offset, reverse=True):
            zinfo._end_offset = end_offset
            end_offset = zinfo.header_offset

    def namelist(self):
        """Return a list of file names in the archive."""
        return [data.filename for data in self.filelist]

    def infolist(self):
        """Return a list of class ZipInfo instances for files in the
        archive."""
        return self.filelist

    def printdir(self, file=None):
        """Print a table of contents for the zip file."""
        print('%-46s %19s %12s' % ('File Name', 'Modified    ', 'Size'), file=file)
        for zinfo in self.filelist:
            date = '%d-%02d-%02d %02d:%02d:%02d' % zinfo.date_time[:6]
            print('%-46s %s %12d' % (zinfo.filename, date, zinfo.file_size), file=file)

    def testzip(self):
        """Read all the files and check the CRC."""
        chunk_size = 2 ** 20
        for zinfo in self.filelist:
            try:
                with self.open(zinfo.filename, 'r') as f:
                    while f.read(chunk_size):
                        pass
            except BadZipFile:
                return zinfo.filename

    def getinfo(self, name):
        """Return the instance of ZipInfo given 'name'."""
        info = self.NameToInfo.get(name)
        if info is None:
            raise KeyError('There is no item named %r in the archive' % name)
        return info

    def setpassword(self, pwd):
        """Set default password for encrypted files."""
        if pwd and (not isinstance(pwd, bytes)):
            raise TypeError('pwd: expected bytes, got %s' % type(pwd).__name__)
        if pwd:
            self.pwd = pwd
        else:
            self.pwd = None

    @property
    def comment(self):
        """The comment text associated with the ZIP file."""
        return self._comment

    @comment.setter
    def comment(self, comment):
        if not isinstance(comment, bytes):
            raise TypeError('comment: expected bytes, got %s' % type(comment).__name__)
        if len(comment) > ZIP_MAX_COMMENT:
            import warnings
            warnings.warn('Archive comment is too long; truncating to %d bytes' % ZIP_MAX_COMMENT, stacklevel=2)
            comment = comment[:ZIP_MAX_COMMENT]
        self._comment = comment
        self._didModify = True

    def read(self, name, pwd=None):
        """Return file bytes for name."""
        with self.open(name, 'r', pwd) as fp:
            return fp.read()

    def open(self, name, mode='r', pwd=None, *, force_zip64=False):
        """Return file-like object for 'name'.

        name is a string for the file name within the ZIP file, or a ZipInfo
        object.

        mode should be 'r' to read a file already in the ZIP file, or 'w' to
        write to a file newly added to the archive.

        pwd is the password to decrypt files (only used for reading).

        When writing, if the file size is not known in advance but may exceed
        2 GiB, pass force_zip64 to use the ZIP64 format, which can handle large
        files.  If the size is known in advance, it is best to pass a ZipInfo
        instance for name, with zinfo.file_size set.
        """
        if mode not in {'r', 'w'}:
            raise ValueError('open() requires mode "r" or "w"')
        if pwd and mode == 'w':
            raise ValueError('pwd is only supported for reading files')
        if not self.fp:
            raise ValueError('Attempt to use ZIP archive that was already closed')
        if isinstance(name, ZipInfo):
            zinfo = name
        elif mode == 'w':
            zinfo = ZipInfo(name)
            zinfo.compress_type = self.compression
            zinfo._compresslevel = self.compresslevel
        else:
            zinfo = self.getinfo(name)
        if mode == 'w':
            return self._open_to_write(zinfo, force_zip64=force_zip64)
        if self._writing:
            raise ValueError("Can't read from the ZIP file while there is an open writing handle on it. Close the writing handle before trying to read.")
        self._fileRefCnt += 1
        zef_file = _SharedFile(self.fp, zinfo.header_offset, self._fpclose, self._lock, lambda: self._writing)
        try:
            fheader = zef_file.read(sizeFileHeader)
            if len(fheader) != sizeFileHeader:
                raise BadZipFile('Truncated file header')
            fheader = struct.unpack(structFileHeader, fheader)
            if fheader[_FH_SIGNATURE] != stringFileHeader:
                raise BadZipFile('Bad magic number for file header')
            fname = zef_file.read(fheader[_FH_FILENAME_LENGTH])
            if fheader[_FH_EXTRA_FIELD_LENGTH]:
                zef_file.read(fheader[_FH_EXTRA_FIELD_LENGTH])
            if zinfo.flag_bits & _MASK_COMPRESSED_PATCH:
                raise NotImplementedError('compressed patched data (flag bit 5)')
            if zinfo.flag_bits & _MASK_STRONG_ENCRYPTION:
                raise NotImplementedError('strong encryption (flag bit 6)')
            if fheader[_FH_GENERAL_PURPOSE_FLAG_BITS] & _MASK_UTF_FILENAME:
                fname_str = fname.decode('utf-8')
            else:
                fname_str = fname.decode(self.metadata_encoding or 'cp437')
            if fname_str != zinfo.orig_filename:
                raise BadZipFile('File name in directory %r and header %r differ.' % (zinfo.orig_filename, fname))
            if zinfo._end_offset is not None and zef_file.tell() + zinfo.compress_size > zinfo._end_offset:
                raise BadZipFile(f'Overlapped entries: {zinfo.orig_filename!r} (possible zip bomb)')
            is_encrypted = zinfo.flag_bits & _MASK_ENCRYPTED
            if is_encrypted:
                if not pwd:
                    pwd = self.pwd
                if pwd and (not isinstance(pwd, bytes)):
                    raise TypeError('pwd: expected bytes, got %s' % type(pwd).__name__)
                if not pwd:
                    raise RuntimeError('File %r is encrypted, password required for extraction' % name)
            else:
                pwd = None
            return ZipExtFile(zef_file, mode, zinfo, pwd, True)
        except:
            zef_file.close()
            raise

    def _open_to_write(self, zinfo, force_zip64=False):
        if force_zip64 and (not self._allowZip64):
            raise ValueError('force_zip64 is True, but allowZip64 was False when opening the ZIP file.')
        if self._writing:
            raise ValueError("Can't write to the ZIP file while there is another write handle open on it. Close the first handle before opening another.")
        zinfo.compress_size = 0
        zinfo.CRC = 0
        zinfo.flag_bits = 0
        if zinfo.compress_type == ZIP_LZMA:
            zinfo.flag_bits |= _MASK_COMPRESS_OPTION_1
        if not self._seekable:
            zinfo.flag_bits |= _MASK_USE_DATA_DESCRIPTOR
        if not zinfo.external_attr:
            zinfo.external_attr = 384 << 16
        zip64 = force_zip64 or zinfo.file_size * 1.05 > ZIP64_LIMIT
        if not self._allowZip64 and zip64:
            raise LargeZipFile('Filesize would require ZIP64 extensions')
        if self._seekable:
            self.fp.seek(self.start_dir)
        zinfo.header_offset = self.fp.tell()
        self._writecheck(zinfo)
        self._didModify = True
        self.fp.write(zinfo.FileHeader(zip64))
        self._writing = True
        return _ZipWriteFile(self, zinfo, zip64)

    def extract(self, member, path=None, pwd=None):
        """Extract a member from the archive to the current working directory,
           using its full name. Its file information is extracted as accurately
           as possible. `member' may be a filename or a ZipInfo object. You can
           specify a different directory using `path'.
        """
        if path is None:
            path = os.getcwd()
        else:
            path = os.fspath(path)
        return self._extract_member(member, path, pwd)

    def extractall(self, path=None, members=None, pwd=None):
        """Extract all members from the archive to the current working
           directory. `path' specifies a different directory to extract to.
           `members' is optional and must be a subset of the list returned
           by namelist().
        """
        if members is None:
            members = self.namelist()
        if path is None:
            path = os.getcwd()
        else:
            path = os.fspath(path)
        for zipinfo in members:
            self._extract_member(zipinfo, path, pwd)

    @classmethod
    def _sanitize_windows_name(cls, arcname, pathsep):
        """Replace bad characters and remove trailing dots from parts."""
        table = cls._windows_illegal_name_trans_table
        if not table:
            illegal = ':<>|"?*'
            table = str.maketrans(illegal, '_' * len(illegal))
            cls._windows_illegal_name_trans_table = table
        arcname = arcname.translate(table)
        arcname = (x.rstrip('.') for x in arcname.split(pathsep))
        arcname = pathsep.join((x for x in arcname if x))
        return arcname

    def _extract_member(self, member, targetpath, pwd):
        """Extract the ZipInfo object 'member' to a physical
           file on the path targetpath.
        """
        if not isinstance(member, ZipInfo):
            member = self.getinfo(member)
        arcname = member.filename.replace('/', os.path.sep)
        if os.path.altsep:
            arcname = arcname.replace(os.path.altsep, os.path.sep)
        arcname = os.path.splitdrive(arcname)[1]
        invalid_path_parts = ('', os.path.curdir, os.path.pardir)
        arcname = os.path.sep.join((x for x in arcname.split(os.path.sep) if x not in invalid_path_parts))
        if os.path.sep == '\\':
            arcname = self._sanitize_windows_name(arcname, os.path.sep)
        targetpath = os.path.join(targetpath, arcname)
        targetpath = os.path.normpath(targetpath)
        upperdirs = os.path.dirname(targetpath)
        if upperdirs and (not os.path.exists(upperdirs)):
            os.makedirs(upperdirs)
        if member.is_dir():
            if not os.path.isdir(targetpath):
                os.mkdir(targetpath)
            return targetpath
        with self.open(member, pwd=pwd) as source, open(targetpath, 'wb') as target:
            shutil.copyfileobj(source, target)
        return targetpath

    def _writecheck(self, zinfo):
        """Check for errors before writing a file to the archive."""
        if zinfo.filename in self.NameToInfo:
            import warnings
            warnings.warn('Duplicate name: %r' % zinfo.filename, stacklevel=3)
        if self.mode not in ('w', 'x', 'a'):
            raise ValueError("write() requires mode 'w', 'x', or 'a'")
        if not self.fp:
            raise ValueError('Attempt to write ZIP archive that was already closed')
        _check_compression(zinfo.compress_type)
        if not self._allowZip64:
            requires_zip64 = None
            if len(self.filelist) >= ZIP_FILECOUNT_LIMIT:
                requires_zip64 = 'Files count'
            elif zinfo.file_size > ZIP64_LIMIT:
                requires_zip64 = 'Filesize'
            elif zinfo.header_offset > ZIP64_LIMIT:
                requires_zip64 = 'Zipfile size'
            if requires_zip64:
                raise LargeZipFile(requires_zip64 + ' would require ZIP64 extensions')

    def write(self, filename, arcname=None, compress_type=None, compresslevel=None):
        """Put the bytes from filename into the archive under the name
        arcname."""
        if not self.fp:
            raise ValueError('Attempt to write to ZIP archive that was already closed')
        if self._writing:
            raise ValueError("Can't write to ZIP archive while an open writing handle exists")
        zinfo = ZipInfo.from_file(filename, arcname, strict_timestamps=self._strict_timestamps)
        if zinfo.is_dir():
            zinfo.compress_size = 0
            zinfo.CRC = 0
            self.mkdir(zinfo)
        else:
            if compress_type is not None:
                zinfo.compress_type = compress_type
            else:
                zinfo.compress_type = self.compression
            if compresslevel is not None:
                zinfo._compresslevel = compresslevel
            else:
                zinfo._compresslevel = self.compresslevel
            with open(filename, 'rb') as src, self.open(zinfo, 'w') as dest:
                shutil.copyfileobj(src, dest, 1024 * 8)

    def writestr(self, zinfo_or_arcname, data, compress_type=None, compresslevel=None):
        """Write a file into the archive.  The contents is 'data', which
        may be either a 'str' or a 'bytes' instance; if it is a 'str',
        it is encoded as UTF-8 first.
        'zinfo_or_arcname' is either a ZipInfo instance or
        the name of the file in the archive."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        if not isinstance(zinfo_or_arcname, ZipInfo):
            zinfo = ZipInfo(filename=zinfo_or_arcname, date_time=time.localtime(time.time())[:6])
            zinfo.compress_type = self.compression
            zinfo._compresslevel = self.compresslevel
            if zinfo.filename[-1] == '/':
                zinfo.external_attr = 16893 << 16
                zinfo.external_attr |= 16
            else:
                zinfo.external_attr = 384 << 16
        else:
            zinfo = zinfo_or_arcname
        if not self.fp:
            raise ValueError('Attempt to write to ZIP archive that was already closed')
        if self._writing:
            raise ValueError("Can't write to ZIP archive while an open writing handle exists.")
        if compress_type is not None:
            zinfo.compress_type = compress_type
        if compresslevel is not None:
            zinfo._compresslevel = compresslevel
        zinfo.file_size = len(data)
        with self._lock:
            with self.open(zinfo, mode='w') as dest:
                dest.write(data)

    def mkdir(self, zinfo_or_directory_name, mode=511):
        """Creates a directory inside the zip archive."""
        if isinstance(zinfo_or_directory_name, ZipInfo):
            zinfo = zinfo_or_directory_name
            if not zinfo.is_dir():
                raise ValueError('The given ZipInfo does not describe a directory')
        elif isinstance(zinfo_or_directory_name, str):
            directory_name = zinfo_or_directory_name
            if not directory_name.endswith('/'):
                directory_name += '/'
            zinfo = ZipInfo(directory_name)
            zinfo.compress_size = 0
            zinfo.CRC = 0
            zinfo.external_attr = ((16384 | mode) & 65535) << 16
            zinfo.file_size = 0
            zinfo.external_attr |= 16
        else:
            raise TypeError('Expected type str or ZipInfo')
        with self._lock:
            if self._seekable:
                self.fp.seek(self.start_dir)
            zinfo.header_offset = self.fp.tell()
            if zinfo.compress_type == ZIP_LZMA:
                zinfo.flag_bits |= _MASK_COMPRESS_OPTION_1
            self._writecheck(zinfo)
            self._didModify = True
            self.filelist.append(zinfo)
            self.NameToInfo[zinfo.filename] = zinfo
            self.fp.write(zinfo.FileHeader(False))
            self.start_dir = self.fp.tell()

    def __del__(self):
        """Call the "close()" method in case the user forgot."""
        self.close()

    def close(self):
        """Close the file, and for mode 'w', 'x' and 'a' write the ending
        records."""
        if self.fp is None:
            return
        if self._writing:
            raise ValueError("Can't close the ZIP file while there is an open writing handle on it. Close the writing handle before closing the zip.")
        try:
            if self.mode in ('w', 'x', 'a') and self._didModify:
                with self._lock:
                    if self._seekable:
                        self.fp.seek(self.start_dir)
                    self._write_end_record()
        finally:
            fp = self.fp
            self.fp = None
            self._fpclose(fp)

    def _write_end_record(self):
        for zinfo in self.filelist:
            dt = zinfo.date_time
            dosdate = dt[0] - 1980 << 9 | dt[1] << 5 | dt[2]
            dostime = dt[3] << 11 | dt[4] << 5 | dt[5] // 2
            extra = []
            if zinfo.file_size > ZIP64_LIMIT or zinfo.compress_size > ZIP64_LIMIT:
                extra.append(zinfo.file_size)
                extra.append(zinfo.compress_size)
                file_size = 4294967295
                compress_size = 4294967295
            else:
                file_size = zinfo.file_size
                compress_size = zinfo.compress_size
            if zinfo.header_offset > ZIP64_LIMIT:
                extra.append(zinfo.header_offset)
                header_offset = 4294967295
            else:
                header_offset = zinfo.header_offset
            extra_data = zinfo.extra
            min_version = 0
            if extra:
                extra_data = _strip_extra(extra_data, (1,))
                extra_data = struct.pack('<HH' + 'Q' * len(extra), 1, 8 * len(extra), *extra) + extra_data
                min_version = ZIP64_VERSION
            if zinfo.compress_type == ZIP_BZIP2:
                min_version = max(BZIP2_VERSION, min_version)
            elif zinfo.compress_type == ZIP_LZMA:
                min_version = max(LZMA_VERSION, min_version)
            extract_version = max(min_version, zinfo.extract_version)
            create_version = max(min_version, zinfo.create_version)
            filename, flag_bits = zinfo._encodeFilenameFlags()
            centdir = struct.pack(structCentralDir, stringCentralDir, create_version, zinfo.create_system, extract_version, zinfo.reserved, flag_bits, zinfo.compress_type, dostime, dosdate, zinfo.CRC, compress_size, file_size, len(filename), len(extra_data), len(zinfo.comment), 0, zinfo.internal_attr, zinfo.external_attr, header_offset)
            self.fp.write(centdir)
            self.fp.write(filename)
            self.fp.write(extra_data)
            self.fp.write(zinfo.comment)
        pos2 = self.fp.tell()
        centDirCount = len(self.filelist)
        centDirSize = pos2 - self.start_dir
        centDirOffset = self.start_dir
        requires_zip64 = None
        if centDirCount > ZIP_FILECOUNT_LIMIT:
            requires_zip64 = 'Files count'
        elif centDirOffset > ZIP64_LIMIT:
            requires_zip64 = 'Central directory offset'
        elif centDirSize > ZIP64_LIMIT:
            requires_zip64 = 'Central directory size'
        if requires_zip64:
            if not self._allowZip64:
                raise LargeZipFile(requires_zip64 + ' would require ZIP64 extensions')
            zip64endrec = struct.pack(structEndArchive64, stringEndArchive64, 44, 45, 45, 0, 0, centDirCount, centDirCount, centDirSize, centDirOffset)
            self.fp.write(zip64endrec)
            zip64locrec = struct.pack(structEndArchive64Locator, stringEndArchive64Locator, 0, pos2, 1)
            self.fp.write(zip64locrec)
            centDirCount = min(centDirCount, 65535)
            centDirSize = min(centDirSize, 4294967295)
            centDirOffset = min(centDirOffset, 4294967295)
        endrec = struct.pack(structEndArchive, stringEndArchive, 0, 0, centDirCount, centDirCount, centDirSize, centDirOffset, len(self._comment))
        self.fp.write(endrec)
        self.fp.write(self._comment)
        if self.mode == 'a':
            self.fp.truncate()
        self.fp.flush()

    def _fpclose(self, fp):
        assert self._fileRefCnt > 0
        self._fileRefCnt -= 1
        if not self._fileRefCnt and (not self._filePassed):
            fp.close()