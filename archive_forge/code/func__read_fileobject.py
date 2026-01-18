import pickle
import io
import sys
import warnings
import contextlib
from .compressor import _ZFILE_PREFIX
from .compressor import _COMPRESSORS
@contextlib.contextmanager
def _read_fileobject(fileobj, filename, mmap_mode=None):
    """Utility function opening the right fileobject from a filename.

    The magic number is used to choose between the type of file object to open:
    * regular file object (default)
    * zlib file object
    * gzip file object
    * bz2 file object
    * lzma file object (for xz and lzma compressor)

    Parameters
    ----------
    fileobj: file object
    compressor: str in {'zlib', 'gzip', 'bz2', 'lzma', 'xz', 'compat',
                        'not-compressed'}
    filename: str
        filename path corresponding to the fileobj parameter.
    mmap_mode: str
        memory map mode that should be used to open the pickle file. This
        parameter is useful to verify that the user is not trying to one with
        compression. Default: None.

    Returns
    -------
        a file like object

    """
    compressor = _detect_compressor(fileobj)
    if compressor == 'compat':
        warnings.warn("The file '%s' has been generated with a joblib version less than 0.10. Please regenerate this pickle file." % filename, DeprecationWarning, stacklevel=2)
        yield filename
    else:
        if compressor in _COMPRESSORS:
            compressor_wrapper = _COMPRESSORS[compressor]
            inst = compressor_wrapper.decompressor_file(fileobj)
            fileobj = _buffered_read_file(inst)
        if mmap_mode is not None:
            if isinstance(fileobj, io.BytesIO):
                warnings.warn('In memory persistence is not compatible with mmap_mode "%(mmap_mode)s" flag passed. mmap_mode option will be ignored.' % locals(), stacklevel=2)
            elif compressor != 'not-compressed':
                warnings.warn('mmap_mode "%(mmap_mode)s" is not compatible with compressed file %(filename)s. "%(mmap_mode)s" flag will be ignored.' % locals(), stacklevel=2)
            elif not _is_raw_file(fileobj):
                warnings.warn('"%(fileobj)r" is not a raw file, mmap_mode "%(mmap_mode)s" flag will be ignored.' % locals(), stacklevel=2)
        yield fileobj