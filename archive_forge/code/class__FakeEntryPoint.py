from __future__ import unicode_literals
import os.path  # splitext
import pkg_resources
from pybtex.exceptions import PybtexError
class _FakeEntryPoint(pkg_resources.EntryPoint):

    def __init__(self, name, klass):
        self.name = name
        self.klass = klass

    def __str__(self):
        return '%s = :%s' % (self.name, self.klass.__name__)

    def __repr__(self):
        return '_FakeEntryPoint(name=%r, klass=%s)' % (self.name, self.klass.__name__)

    def load(self, require=True, env=None, installer=None):
        return self.klass

    def require(self, env=None, installer=None):
        pass