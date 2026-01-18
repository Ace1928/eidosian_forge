import os, sys, time, re
import py
from py import path, process
from py._path import common
from py._path import svnwc as svncommon
from py._path.cacheutil import BuildcostAccessCache, AgingCache
def _norev_delentry(self, path):
    auth = self.auth and self.auth.makecmdoptions() or None
    self._lsnorevcache.delentry((str(path), auth))