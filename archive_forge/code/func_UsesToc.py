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
def UsesToc(self, flavor):
    """Return true if the target should produce a restat rule based on a TOC
        file."""
    if flavor == 'win' or self.bundle:
        return False
    return self.type in ('shared_library', 'loadable_module')