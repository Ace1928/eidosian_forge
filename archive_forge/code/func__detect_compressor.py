import pickle
import io
import sys
import warnings
import contextlib
from .compressor import _ZFILE_PREFIX
from .compressor import _COMPRESSORS
def _detect_compressor(fileobj):
    """Return the compressor matching fileobj.

    Parameters
    ----------
    fileobj: file object

    Returns
    -------
    str in {'zlib', 'gzip', 'bz2', 'lzma', 'xz', 'compat', 'not-compressed'}
    """
    max_prefix_len = _get_prefixes_max_len()
    if hasattr(fileobj, 'peek'):
        first_bytes = fileobj.peek(max_prefix_len)
    else:
        first_bytes = fileobj.read(max_prefix_len)
        fileobj.seek(0)
    if first_bytes.startswith(_ZFILE_PREFIX):
        return 'compat'
    else:
        for name, compressor in _COMPRESSORS.items():
            if first_bytes.startswith(compressor.prefix):
                return name
    return 'not-compressed'