import os, sys, time, re
import py
from py import path, process
from py._path import common
from py._path import svnwc as svncommon
from py._path.cacheutil import BuildcostAccessCache, AgingCache
def _svncmdexecauth(self, cmd):
    """ execute an svn command 'as is' """
    cmd = svncommon.fixlocale() + cmd
    if self.auth is not None:
        cmd += ' ' + self.auth.makecmdoptions()
    return self._cmdexec(cmd)