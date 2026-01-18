import ctypes
import logging
import os
import platform
import shutil
import stat
import sys
import tempfile
import subprocess
from io import StringIO
import pyomo.common.unittest as unittest
import pyomo.common.envvar as envvar
from pyomo.common.log import LoggingIntercept
from pyomo.common.fileutils import (
from pyomo.common.download import FileDownloader
def _make_exec(self, fname):
    open(fname, 'w').close()
    mode = os.stat(fname).st_mode
    os.chmod(fname, mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)