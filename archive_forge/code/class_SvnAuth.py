import os, sys, time, re, calendar
import py
import subprocess
from py._path import common
class SvnAuth(object):
    """ container for auth information for Subversion """

    def __init__(self, username, password, cache_auth=True, interactive=True):
        self.username = username
        self.password = password
        self.cache_auth = cache_auth
        self.interactive = interactive

    def makecmdoptions(self):
        uname = self.username.replace('"', '\\"')
        passwd = self.password.replace('"', '\\"')
        ret = []
        if uname:
            ret.append('--username="%s"' % (uname,))
        if passwd:
            ret.append('--password="%s"' % (passwd,))
        if not self.cache_auth:
            ret.append('--no-auth-cache')
        if not self.interactive:
            ret.append('--non-interactive')
        return ' '.join(ret)

    def __str__(self):
        return '<SvnAuth username=%s ...>' % (self.username,)