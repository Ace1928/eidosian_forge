import os
import sys
import stat
import fnmatch
import collections
import errno
def _unpack_tarfile(filename, extract_dir, *, filter=None):
    """Unpack tar/tar.gz/tar.bz2/tar.xz `filename` to `extract_dir`
    """
    import tarfile
    try:
        tarobj = tarfile.open(filename)
    except tarfile.TarError:
        raise ReadError('%s is not a compressed or uncompressed tar file' % filename)
    try:
        tarobj.extractall(extract_dir, filter=filter)
    finally:
        tarobj.close()