from __future__ import absolute_import
import re
import sys
import os
import tokenize
from io import StringIO
from ._looper import looper
from .compat3 import bytes, unicode_, basestring_, next, is_unicode, coerce_text
class bunch(dict):

    def __init__(self, **kw):
        for name, value in kw.items():
            setattr(self, name, value)

    def __setattr__(self, name, value):
        self[name] = value

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __getitem__(self, key):
        if 'default' in self:
            try:
                return dict.__getitem__(self, key)
            except KeyError:
                return dict.__getitem__(self, 'default')
        else:
            return dict.__getitem__(self, key)

    def __repr__(self):
        return '<%s %s>' % (self.__class__.__name__, ' '.join(['%s=%r' % (k, v) for k, v in sorted(self.items())]))