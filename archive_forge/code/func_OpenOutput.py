import collections
import copy
import hashlib
import json
import multiprocessing
import os.path
import re
import signal
import subprocess
import sys
import gyp
import gyp.common
import gyp.msvs_emulation
import gyp.MSVSUtil as MSVSUtil
import gyp.xcode_emulation
from io import StringIO
from gyp.common import GetEnvironFallback
import gyp.ninja_syntax as ninja_syntax
def OpenOutput(path, mode='w'):
    """Open |path| for writing, creating directories if necessary."""
    gyp.common.EnsureDirExists(path)
    return open(path, mode)