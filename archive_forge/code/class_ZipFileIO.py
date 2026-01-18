import os
from parso import file_io
class ZipFileIO(file_io.KnownContentFileIO, FileIOFolderMixin):
    """For .zip and .egg archives"""

    def __init__(self, path, code, zip_path):
        super().__init__(path, code)
        self._zip_path = zip_path

    def get_last_modified(self):
        try:
            return os.path.getmtime(self._zip_path)
        except (FileNotFoundError, PermissionError, NotADirectoryError):
            return None