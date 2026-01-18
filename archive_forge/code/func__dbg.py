from builtins import open as bltn_open
import sys
import os
import io
import shutil
import stat
import time
import struct
import copy
import re
import warnings
def _dbg(self, level, msg):
    """Write debugging output to sys.stderr.
        """
    if level <= self.debug:
        print(msg, file=sys.stderr)