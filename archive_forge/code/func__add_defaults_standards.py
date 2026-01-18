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
def _add_defaults_standards(self):
    standards = [self.READMES, self.distribution.script_name]
    for fn in standards:
        if isinstance(fn, tuple):
            alts = fn
            got_it = False
            for fn in alts:
                if self._cs_path_exists(fn):
                    got_it = True
                    self.filelist.append(fn)
                    break
            if not got_it:
                self.warn('standard file not found: should have one of ' + ', '.join(alts))
        elif self._cs_path_exists(fn):
            self.filelist.append(fn)
        else:
            self.warn("standard file '%s' not found" % fn)