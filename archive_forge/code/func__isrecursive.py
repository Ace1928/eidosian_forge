import contextlib
import os
import re
import fnmatch
import itertools
import stat
import sys
def _isrecursive(pattern):
    if isinstance(pattern, bytes):
        return pattern == b'**'
    else:
        return pattern == '**'