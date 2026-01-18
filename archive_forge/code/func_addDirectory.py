import sys, os, pickle
from hashlib import md5
from xml.sax.saxutils import quoteattr
from time import process_time as clock
from reportlab.lib.utils import asBytes, asNative as _asNative
from reportlab.lib.utils import rl_isdir, rl_isfile, rl_listdir, rl_getmtime
def addDirectory(self, dirName, recur=None):
    if rl_isdir(dirName):
        self._dirs.add(dirName)
        if recur if recur is not None else self._recur:
            for r, D, F in os.walk(dirName):
                for d in D:
                    self._dirs.add(os.path.join(r, d))