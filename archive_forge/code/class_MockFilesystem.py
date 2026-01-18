from __future__ import unicode_literals
import errno
import posixpath
from unittest import TestCase
from pybtex import io
class MockFilesystem(object):

    def __init__(self, files=(), writable_dirs=(), readonly_dirs=()):
        self.files = set(files)
        self.writable_dirs = set(writable_dirs)
        self.readonly_dirs = set(readonly_dirs)

    def add_file(self, path):
        self.files.add(path)

    def chdir(self, path):
        self.pwd = path

    def locate(self, filename):
        for path in self.files:
            if path.endswith(filename):
                return path

    def open_read(self, path, mode):
        if path in self.files:
            return MockFile(path, mode)
        else:
            raise IOError(errno.ENOENT, 'No such file or directory', path)

    def open_write(self, path, mode):
        dirname = posixpath.dirname(path)
        if dirname in self.writable_dirs:
            return MockFile(path, mode)
        else:
            raise IOError(errno.EACCES, 'Permission denied', path)

    def open(self, path, mode):
        full_path = posixpath.join(self.pwd, path)
        if 'w' in mode:
            return self.open_write(full_path, mode)
        else:
            return self.open_read(full_path, mode)