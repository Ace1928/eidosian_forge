import os
import shutil
import stat
import tarfile
import zipfile
from django.core.exceptions import SuspiciousOperation
@staticmethod
def _archive_cls(file):
    cls = None
    if isinstance(file, str):
        filename = file
    else:
        try:
            filename = file.name
        except AttributeError:
            raise UnrecognizedArchiveFormat('File object not a recognized archive format.')
    base, tail_ext = os.path.splitext(filename.lower())
    cls = extension_map.get(tail_ext)
    if not cls:
        base, ext = os.path.splitext(base)
        cls = extension_map.get(ext)
    if not cls:
        raise UnrecognizedArchiveFormat('Path not a recognized archive format: %s' % filename)
    return cls