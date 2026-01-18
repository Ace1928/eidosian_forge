import os, sys, time, re, calendar
import py
import subprocess
from py._path import common
def _svn(self, cmd, *args):
    l = ['svn %s' % cmd]
    args = [self._escape(item) for item in args]
    l.extend(args)
    l.append('"%s"' % self._escape(self.strpath))
    string = fixlocale() + ' '.join(l)
    try:
        try:
            key = 'LC_MESSAGES'
            hold = os.environ.get(key)
            os.environ[key] = 'C'
            out = py.process.cmdexec(string)
        finally:
            if hold:
                os.environ[key] = hold
            else:
                del os.environ[key]
    except py.process.cmdexec.Error:
        e = sys.exc_info()[1]
        strerr = e.err.lower()
        if strerr.find('not found') != -1:
            raise py.error.ENOENT(self)
        elif strerr.find('E200009:') != -1:
            raise py.error.ENOENT(self)
        if strerr.find('file exists') != -1 or strerr.find('file already exists') != -1 or strerr.find('w150002:') != -1 or (strerr.find("can't create directory") != -1):
            raise py.error.EEXIST(strerr)
        raise
    return out