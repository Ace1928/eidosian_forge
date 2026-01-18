from math import log
import os
from os import path as op
import sys
import shutil
import time
from . import appdata_dir, resource_dirs
from . import StdoutProgressIndicator, urlopen
class InternetNotAllowedError(IOError):
    """Plugins that need resources can just use get_remote_file(), but
    should catch this error and silently ignore it.
    """
    pass