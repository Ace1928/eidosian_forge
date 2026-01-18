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