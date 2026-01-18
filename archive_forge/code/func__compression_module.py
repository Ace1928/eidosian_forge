import abc
import os
import bz2
import gzip
import lzma
import shutil
from zipfile import ZipFile
from tarfile import TarFile
from .utils import get_logger
def _compression_module(self, fname):
    """
        Get the Python module compatible with fname and the chosen method.

        If the *method* attribute is "auto", will select a method based on the
        extension. If no recognized extension is in the file name, will raise a
        ValueError.
        """
    error_archives = 'To unpack zip/tar archives, use pooch.Unzip/Untar instead.'
    if self.method not in self.modules:
        message = f"Invalid compression method '{self.method}'. Must be one of '{list(self.modules.keys())}'."
        if self.method in {'zip', 'tar'}:
            message = ' '.join([message, error_archives])
        raise ValueError(message)
    if self.method == 'auto':
        ext = os.path.splitext(fname)[-1]
        if ext not in self.extensions:
            message = f"Unrecognized file extension '{ext}'. Must be one of '{list(self.extensions.keys())}'."
            if ext in {'.zip', '.tar'}:
                message = ' '.join([message, error_archives])
            raise ValueError(message)
        return self.modules[self.extensions[ext]]
    return self.modules[self.method]