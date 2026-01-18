import io
import os
import numpy as np
import scipy.sparse
from scipy.io import _mmio
def _get_read_cursor(source, parallelism=None):
    """
    Open file for reading.
    """
    from . import _fmm_core
    ret_stream_to_close = None
    if parallelism is None:
        parallelism = PARALLELISM
    try:
        source = os.fspath(source)
        is_path = True
    except TypeError:
        is_path = False
    if is_path:
        path = str(source)
        if path.endswith('.gz'):
            import gzip
            source = gzip.GzipFile(path, 'r')
            ret_stream_to_close = source
        elif path.endswith('.bz2'):
            import bz2
            source = bz2.BZ2File(path, 'rb')
            ret_stream_to_close = source
        else:
            return (_fmm_core.open_read_file(path, parallelism), ret_stream_to_close)
    if hasattr(source, 'read'):
        if isinstance(source, io.TextIOBase):
            source = _TextToBytesWrapper(source)
        return (_fmm_core.open_read_stream(source, parallelism), ret_stream_to_close)
    else:
        raise TypeError('Unknown source type')