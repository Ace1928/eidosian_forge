import sys
import io
import random
import mimetypes
import time
import os
import shutil
import smtplib
import shlex
import re
import subprocess
from urllib.parse import urlencode
from urllib import parse as urlparse
from http.cookies import BaseCookie
from paste import wsgilib
from paste import lint
from paste.response import HeaderDict
class FoundFile(object):
    """
    Represents a single file found as the result of a command.

    Has attributes:

    ``path``:
        The path of the file, relative to the ``base_path``

    ``full``:
        The full path

    ``stat``:
        The results of ``os.stat``.  Also ``mtime`` and ``size``
        contain the ``.st_mtime`` and ``st_size`` of the stat.

    ``bytes``:
        The contents of the file.

    You may use the ``in`` operator with these objects (tested against
    the contents of the file), and the ``.mustcontain()`` method.
    """
    file = True
    dir = False

    def __init__(self, base_path, path):
        self.base_path = base_path
        self.path = path
        self.full = os.path.join(base_path, path)
        self.stat = os.stat(self.full)
        self.mtime = self.stat.st_mtime
        self.size = self.stat.st_size
        self._bytes = None

    def bytes__get(self):
        if self._bytes is None:
            f = open(self.full, 'rb')
            self._bytes = f.read()
            f.close()
        return self._bytes
    bytes = property(bytes__get)

    def __contains__(self, s):
        return s in self.bytes

    def mustcontain(self, s):
        __tracebackhide__ = True
        bytes_ = self.bytes
        if s not in bytes_:
            print('Could not find %r in:' % s)
            print(bytes_)
            assert s in bytes_

    def __repr__(self):
        return '<%s %s:%s>' % (self.__class__.__name__, self.base_path, self.path)