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
def _add_defaults_python(self):
    build_py = self.get_finalized_command('build_py')
    if self.distribution.has_pure_modules():
        self.filelist.extend(build_py.get_source_files())
    for pkg, src_dir, build_dir, filenames in build_py.data_files:
        for filename in filenames:
            self.filelist.append(os.path.join(src_dir, filename))