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
def _add_defaults_ext(self):
    if self.distribution.has_ext_modules():
        build_ext = self.get_finalized_command('build_ext')
        self.filelist.extend(build_ext.get_source_files())