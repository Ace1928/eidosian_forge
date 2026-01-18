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