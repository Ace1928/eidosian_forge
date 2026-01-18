import os
import sys
from glob import glob
from warnings import warn
from distutils.core import Command
from distutils import dir_util
from distutils import file_util
from distutils import archive_util
from distutils.text_file import TextFile
from distutils.filelist import FileList
from distutils import log
from distutils.util import convert_path
from distutils.errors import DistutilsTemplateError, DistutilsOptionError
@staticmethod
def _cs_path_exists(fspath):
    """
        Case-sensitive path existence check

        >>> sdist._cs_path_exists(__file__)
        True
        >>> sdist._cs_path_exists(__file__.upper())
        False
        """
    if not os.path.exists(fspath):
        return False
    abspath = os.path.abspath(fspath)
    directory, filename = os.path.split(abspath)
    return filename in os.listdir(directory)