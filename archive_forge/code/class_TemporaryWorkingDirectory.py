import os as _os
from pathlib import Path
from tempfile import TemporaryDirectory
class TemporaryWorkingDirectory(TemporaryDirectory):
    """
    Creates a temporary directory and sets the cwd to that directory.
    Automatically reverts to previous cwd upon cleanup.
    Usage example:

        with TemporaryWorkingDirectory() as tmpdir:
            ...
    """

    def __enter__(self):
        self.old_wd = Path.cwd()
        _os.chdir(self.name)
        return super(TemporaryWorkingDirectory, self).__enter__()

    def __exit__(self, exc, value, tb):
        _os.chdir(self.old_wd)
        return super(TemporaryWorkingDirectory, self).__exit__(exc, value, tb)