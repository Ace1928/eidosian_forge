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
def _add_defaults_data_files(self):
    if self.distribution.has_data_files():
        for item in self.distribution.data_files:
            if isinstance(item, str):
                item = convert_path(item)
                if os.path.isfile(item):
                    self.filelist.append(item)
            else:
                dirname, filenames = item
                for f in filenames:
                    f = convert_path(f)
                    if os.path.isfile(f):
                        self.filelist.append(f)