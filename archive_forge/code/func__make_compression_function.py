import datetime
import decimal
import glob
import numbers
import os
import shutil
import string
from functools import partial
from stat import ST_DEV, ST_INO
from . import _string_parsers as string_parsers
from ._ctime_functions import get_ctime, set_ctime
from ._datetime import aware_now
@staticmethod
def _make_compression_function(compression):
    if compression is None:
        return None
    elif isinstance(compression, str):
        ext = compression.strip().lstrip('.')
        if ext == 'gz':
            import gzip
            compress = partial(Compression.copy_compress, opener=gzip.open, mode='wb')
        elif ext == 'bz2':
            import bz2
            compress = partial(Compression.copy_compress, opener=bz2.open, mode='wb')
        elif ext == 'xz':
            import lzma
            compress = partial(Compression.copy_compress, opener=lzma.open, mode='wb', format=lzma.FORMAT_XZ)
        elif ext == 'lzma':
            import lzma
            compress = partial(Compression.copy_compress, opener=lzma.open, mode='wb', format=lzma.FORMAT_ALONE)
        elif ext == 'tar':
            import tarfile
            compress = partial(Compression.add_compress, opener=tarfile.open, mode='w:')
        elif ext == 'tar.gz':
            import gzip
            import tarfile
            compress = partial(Compression.add_compress, opener=tarfile.open, mode='w:gz')
        elif ext == 'tar.bz2':
            import bz2
            import tarfile
            compress = partial(Compression.add_compress, opener=tarfile.open, mode='w:bz2')
        elif ext == 'tar.xz':
            import lzma
            import tarfile
            compress = partial(Compression.add_compress, opener=tarfile.open, mode='w:xz')
        elif ext == 'zip':
            import zipfile
            compress = partial(Compression.write_compress, opener=zipfile.ZipFile, mode='w', compression=zipfile.ZIP_DEFLATED)
        else:
            raise ValueError("Invalid compression format: '%s'" % ext)
        return partial(Compression.compression, ext='.' + ext, compress_function=compress)
    elif callable(compression):
        return compression
    else:
        raise TypeError("Cannot infer compression for objects of type: '%s'" % type(compression).__name__)