from math import log
import os
from os import path as op
import sys
import shutil
import time
from . import appdata_dir, resource_dirs
from . import StdoutProgressIndicator, urlopen
class NeedDownloadError(IOError):
    """Is raised when a remote file is requested that is not locally
    available, but which needs to be explicitly downloaded by the user.
    """