import os
import shutil
import stat
import tarfile
import zipfile
from django.core.exceptions import SuspiciousOperation
class BaseArchive:
    """
    Base Archive class.  Implementations should inherit this class.
    """

    @staticmethod
    def _copy_permissions(mode, filename):
        """
        If the file in the archive has some permissions (this assumes a file
        won't be writable/executable without being readable), apply those
        permissions to the unarchived file.
        """
        if mode & stat.S_IROTH:
            os.chmod(filename, mode)

    def split_leading_dir(self, path):
        path = str(path)
        path = path.lstrip('/').lstrip('\\')
        if '/' in path and ('\\' in path and path.find('/') < path.find('\\') or '\\' not in path):
            return path.split('/', 1)
        elif '\\' in path:
            return path.split('\\', 1)
        else:
            return (path, '')

    def has_leading_dir(self, paths):
        """
        Return True if all the paths have the same leading path name
        (i.e., everything is in one subdirectory in an archive).
        """
        common_prefix = None
        for path in paths:
            prefix, rest = self.split_leading_dir(path)
            if not prefix:
                return False
            elif common_prefix is None:
                common_prefix = prefix
            elif prefix != common_prefix:
                return False
        return True

    def target_filename(self, to_path, name):
        target_path = os.path.abspath(to_path)
        filename = os.path.abspath(os.path.join(target_path, name))
        if not filename.startswith(target_path):
            raise SuspiciousOperation("Archive contains invalid path: '%s'" % name)
        return filename

    def extract(self):
        raise NotImplementedError('subclasses of BaseArchive must provide an extract() method')

    def list(self):
        raise NotImplementedError('subclasses of BaseArchive must provide a list() method')