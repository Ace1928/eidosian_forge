import os
import signal
import subprocess
import sys
import textwrap
import warnings
from distutils.command.config import config as old_config
from distutils.command.config import LANG_EXT
from distutils import log
from distutils.file_util import copy_file
from distutils.ccompiler import CompileError, LinkError
import distutils
from numpy.distutils.exec_command import filepath_from_subprocess_output
from numpy.distutils.mingw32ccompiler import generate_manifest
from numpy.distutils.command.autodist import (check_gcc_function_attribute,
class GrabStdout:

    def __init__(self):
        self.sys_stdout = sys.stdout
        self.data = ''
        sys.stdout = self

    def write(self, data):
        self.sys_stdout.write(data)
        self.data += data

    def flush(self):
        self.sys_stdout.flush()

    def restore(self):
        sys.stdout = self.sys_stdout