import os, sys, time, re
import py
from py import path, process
from py._path import common
from py._path import svnwc as svncommon
from py._path.cacheutil import BuildcostAccessCache, AgingCache
def _listdir_nameinfo(self):
    """ return sequence of name-info directory entries of self """

    def builder():
        try:
            res = self._svnwithrev('ls', '-v')
        except process.cmdexec.Error:
            e = sys.exc_info()[1]
            if e.err.find('non-existent in that revision') != -1:
                raise py.error.ENOENT(self, e.err)
            elif e.err.find('E200009:') != -1:
                raise py.error.ENOENT(self, e.err)
            elif e.err.find('File not found') != -1:
                raise py.error.ENOENT(self, e.err)
            elif e.err.find('not part of a repository') != -1:
                raise py.error.ENOENT(self, e.err)
            elif e.err.find('Unable to open') != -1:
                raise py.error.ENOENT(self, e.err)
            elif e.err.lower().find('method not allowed') != -1:
                raise py.error.EACCES(self, e.err)
            raise py.error.Error(e.err)
        lines = res.split('\n')
        nameinfo_seq = []
        for lsline in lines:
            if lsline:
                info = InfoSvnCommand(lsline)
                if info._name != '.':
                    nameinfo_seq.append((info._name, info))
        nameinfo_seq.sort()
        return nameinfo_seq
    auth = self.auth and self.auth.makecmdoptions() or None
    if self.rev is not None:
        return self._lsrevcache.getorbuild((self.strpath, self.rev, auth), builder)
    else:
        return self._lsnorevcache.getorbuild((self.strpath, auth), builder)