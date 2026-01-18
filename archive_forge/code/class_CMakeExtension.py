import errno
import os
import shutil
import stat
import sys
import tempfile
import pyomo.common.envvar as envvar
from pyomo.common.fileutils import this_file_dir, find_executable
class CMakeExtension(Extension, object):

    def __init__(self, target_dir, user_args, parallel):
        super(CMakeExtension, self).__init__(self.__class__.__qualname__, sources=[])
        self.target_dir = target_dir
        self.user_args = user_args
        self.parallel = parallel