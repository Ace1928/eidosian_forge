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
def PreActionInput(self, flavor):
    """Return the path, if any, that should be used as a dependency of
        any dependent action step."""
    if self.UsesToc(flavor):
        return self.FinalOutput() + '.TOC'
    return self.FinalOutput() or self.preaction_stamp