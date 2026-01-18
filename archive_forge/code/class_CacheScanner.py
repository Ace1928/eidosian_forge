import os
import traceback
from io import StringIO
from twisted import copyright
from twisted.python.compat import execfile, networkString
from twisted.python.filepath import _coerceToFilesystemEncoding
from twisted.web import http, resource, server, static, util
import mygreatresource
class CacheScanner:

    def __init__(self, path, registry):
        self.path = path
        self.registry = registry
        self.doCache = 0

    def cache(self):
        c = self.registry.getCachedPath(self.path)
        if c is not None:
            raise AlreadyCached(c)
        self.recache()

    def recache(self):
        self.doCache = 1