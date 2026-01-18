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
def QuoteShellArgument(arg, flavor):
    """Quote a string such that it will be interpreted as a single argument
    by the shell."""
    if re.match('^[a-zA-Z0-9_=.\\\\/-]+$', arg):
        return arg
    if flavor == 'win':
        return gyp.msvs_emulation.QuoteForRspFile(arg)
    return "'" + arg.replace("'", "'" + '"\'"' + "'") + "'"