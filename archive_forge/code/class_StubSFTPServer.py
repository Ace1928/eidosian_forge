import os
import socket
import socketserver
import sys
import time
import paramiko
from .. import osutils, trace, urlutils
from ..transport import ssh
from . import test_server
class StubSFTPServer(paramiko.SFTPServerInterface):

    def __init__(self, server, root, home=None):
        paramiko.SFTPServerInterface.__init__(self, server)
        self.root = root
        if home is None:
            self.home = ''
        else:
            if not home.startswith(self.root):
                raise AssertionError('home must be a subdirectory of root (%s vs %s)' % (home, root))
            self.home = home[len(self.root):]
        if self.home.startswith('/'):
            self.home = self.home[1:]
        server.log('sftpserver - new connection')

    def _realpath(self, path):
        if self.root == '/':
            return self.canonicalize(path)
        return self.root + self.canonicalize(path)
    if sys.platform == 'win32':

        def canonicalize(self, path):
            thispath = path.decode('utf8')
            if path.startswith('/'):
                return os.path.normpath(thispath[1:])
            else:
                return os.path.normpath(os.path.join(self.home, thispath))
    else:

        def canonicalize(self, path):
            if os.path.isabs(path):
                return osutils.normpath(path)
            else:
                return osutils.normpath('/' + os.path.join(self.home, path))

    def chattr(self, path, attr):
        try:
            paramiko.SFTPServer.set_file_attr(path, attr)
        except OSError as e:
            return paramiko.SFTPServer.convert_errno(e.errno)
        return paramiko.SFTP_OK

    def list_folder(self, path):
        path = self._realpath(path)
        try:
            out = []
            if sys.platform == 'win32':
                flist = [f.encode('utf8') for f in os.listdir(path)]
            else:
                flist = os.listdir(path)
            for fname in flist:
                attr = paramiko.SFTPAttributes.from_stat(os.stat(osutils.pathjoin(path, fname)))
                attr.filename = fname
                out.append(attr)
            return out
        except OSError as e:
            return paramiko.SFTPServer.convert_errno(e.errno)

    def stat(self, path):
        path = self._realpath(path)
        try:
            return paramiko.SFTPAttributes.from_stat(os.stat(path))
        except OSError as e:
            return paramiko.SFTPServer.convert_errno(e.errno)

    def lstat(self, path):
        path = self._realpath(path)
        try:
            return paramiko.SFTPAttributes.from_stat(os.lstat(path))
        except OSError as e:
            return paramiko.SFTPServer.convert_errno(e.errno)

    def open(self, path, flags, attr):
        path = self._realpath(path)
        try:
            flags |= getattr(os, 'O_BINARY', 0)
            if getattr(attr, 'st_mode', None):
                fd = os.open(path, flags, attr.st_mode)
            else:
                fd = os.open(path, flags, 438)
        except OSError as e:
            return paramiko.SFTPServer.convert_errno(e.errno)
        if flags & os.O_CREAT and attr is not None:
            attr._flags &= ~attr.FLAG_PERMISSIONS
            paramiko.SFTPServer.set_file_attr(path, attr)
        if flags & os.O_WRONLY:
            fstr = 'wb'
        elif flags & os.O_RDWR:
            fstr = 'rb+'
        else:
            fstr = 'rb'
        try:
            f = os.fdopen(fd, fstr)
        except OSError as e:
            return paramiko.SFTPServer.convert_errno(e.errno)
        fobj = StubSFTPHandle()
        fobj.filename = path
        fobj.readfile = f
        fobj.writefile = f
        return fobj

    def remove(self, path):
        path = self._realpath(path)
        try:
            os.remove(path)
        except OSError as e:
            return paramiko.SFTPServer.convert_errno(e.errno)
        return paramiko.SFTP_OK

    def rename(self, oldpath, newpath):
        oldpath = self._realpath(oldpath)
        newpath = self._realpath(newpath)
        try:
            os.rename(oldpath, newpath)
        except OSError as e:
            return paramiko.SFTPServer.convert_errno(e.errno)
        return paramiko.SFTP_OK

    def symlink(self, target_path, path):
        path = self._realpath(path)
        try:
            os.symlink(target_path, path)
        except OSError as e:
            return paramiko.SFTPServer.convert_errno(e.errno)
        return paramiko.SFTP_OK

    def readlink(self, path):
        path = self._realpath(path)
        try:
            target_path = os.readlink(path)
        except OSError as e:
            return paramiko.SFTPServer.convert_errno(e.errno)
        return target_path

    def mkdir(self, path, attr):
        path = self._realpath(path)
        try:
            if getattr(attr, 'st_mode', None):
                os.mkdir(path, attr.st_mode)
            else:
                os.mkdir(path)
            if attr is not None:
                attr._flags &= ~attr.FLAG_PERMISSIONS
                paramiko.SFTPServer.set_file_attr(path, attr)
        except OSError as e:
            return paramiko.SFTPServer.convert_errno(e.errno)
        return paramiko.SFTP_OK

    def rmdir(self, path):
        path = self._realpath(path)
        try:
            os.rmdir(path)
        except OSError as e:
            return paramiko.SFTPServer.convert_errno(e.errno)
        return paramiko.SFTP_OK