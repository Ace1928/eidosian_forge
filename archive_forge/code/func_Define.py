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
def Define(d, flavor):
    """Takes a preprocessor define and returns a -D parameter that's ninja- and
    shell-escaped."""
    if flavor == 'win':
        d = d.replace('#', '\\%03o' % ord('#'))
    return QuoteShellArgument(ninja_syntax.escape('-D' + d), flavor)