import os
import shutil
import stat
import tarfile
import zipfile
from django.core.exceptions import SuspiciousOperation
@staticmethod
def _copy_permissions(mode, filename):
    """
        If the file in the archive has some permissions (this assumes a file
        won't be writable/executable without being readable), apply those
        permissions to the unarchived file.
        """
    if mode & stat.S_IROTH:
        os.chmod(filename, mode)