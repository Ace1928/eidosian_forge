import os, sys, time, re
import py
from py import path, process
from py._path import common
from py._path import svnwc as svncommon
from py._path.cacheutil import BuildcostAccessCache, AgingCache
def _svnwithrev(self, cmd, *args):
    """ execute an svn command, append our own url and revision """
    if self.rev is None:
        return self._svnwrite(cmd, *args)
    else:
        args = ['-r', self.rev] + list(args)
        return self._svnwrite(cmd, *args)