from __future__ import with_statement
import logging
import optparse
import os
import os.path
import re
import shutil
import subprocess
import sys
import itertools
def _dirmatch(path, matchwith):
    """Check if path is within matchwith's tree.
    >>> _dirmatch('/home/foo/bar', '/home/foo/bar')
    True
    >>> _dirmatch('/home/foo/bar/', '/home/foo/bar')
    True
    >>> _dirmatch('/home/foo/bar/etc', '/home/foo/bar')
    True
    >>> _dirmatch('/home/foo/bar2', '/home/foo/bar')
    False
    >>> _dirmatch('/home/foo/bar2/etc', '/home/foo/bar')
    False
    """
    matchlen = len(matchwith)
    if path.startswith(matchwith) and path[matchlen:matchlen + 1] in [os.sep, '']:
        return True
    return False