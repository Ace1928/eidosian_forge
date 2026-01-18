import os
import posixpath
import sys
import urllib.parse
import warnings
from os.path import join as pjoin
import pyarrow as pa
from pyarrow.util import doc, _stringify_path, _is_path_like, _DEPR_MSG
class S3FSWrapper(DaskFileSystem):

    @doc(FileSystem.isdir)
    def isdir(self, path):
        path = _sanitize_s3(_stringify_path(path))
        try:
            contents = self.fs.ls(path)
            if len(contents) == 1 and contents[0] == path:
                return False
            else:
                return True
        except OSError:
            return False

    @doc(FileSystem.isfile)
    def isfile(self, path):
        path = _sanitize_s3(_stringify_path(path))
        try:
            contents = self.fs.ls(path)
            return len(contents) == 1 and contents[0] == path
        except OSError:
            return False

    def walk(self, path, refresh=False):
        """
        Directory tree generator, like os.walk.

        Generator version of what is in s3fs, which yields a flattened list of
        files.
        """
        path = _sanitize_s3(_stringify_path(path))
        directories = set()
        files = set()
        for key in list(self.fs._ls(path, refresh=refresh)):
            path = key['Key']
            if key['StorageClass'] == 'DIRECTORY':
                directories.add(path)
            elif key['StorageClass'] == 'BUCKET':
                pass
            else:
                files.add(path)
        files = sorted([posixpath.split(f)[1] for f in files if f not in directories])
        directories = sorted([posixpath.split(x)[1] for x in directories])
        yield (path, directories, files)
        for directory in directories:
            yield from self.walk(directory, refresh=refresh)