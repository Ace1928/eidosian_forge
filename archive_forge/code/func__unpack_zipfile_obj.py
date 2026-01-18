import zipfile
import tarfile
import os
import shutil
import posixpath
import contextlib
from distutils.errors import DistutilsError
from ._path import ensure_directory
def _unpack_zipfile_obj(zipfile_obj, extract_dir, progress_filter=default_filter):
    """Internal/private API used by other parts of setuptools.
    Similar to ``unpack_zipfile``, but receives an already opened :obj:`zipfile.ZipFile`
    object instead of a filename.
    """
    for info in zipfile_obj.infolist():
        name = info.filename
        if name.startswith('/') or '..' in name.split('/'):
            continue
        target = os.path.join(extract_dir, *name.split('/'))
        target = progress_filter(name, target)
        if not target:
            continue
        if name.endswith('/'):
            ensure_directory(target)
        else:
            ensure_directory(target)
            data = zipfile_obj.read(info.filename)
            with open(target, 'wb') as f:
                f.write(data)
        unix_attributes = info.external_attr >> 16
        if unix_attributes:
            os.chmod(target, unix_attributes)