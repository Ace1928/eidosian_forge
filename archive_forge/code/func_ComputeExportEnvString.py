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
def ComputeExportEnvString(self, env):
    """Given an environment, returns a string looking like
            'export FOO=foo; export BAR="${FOO} bar;'
        that exports |env| to the shell."""
    export_str = []
    for k, v in env:
        export_str.append('export %s=%s;' % (k, ninja_syntax.escape(gyp.common.EncodePOSIXShellArgument(v))))
    return ' '.join(export_str)