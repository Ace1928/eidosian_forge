import atexit
import binascii
import collections
import glob
import inspect
import io
import math
import os
import pathlib
import re
import string
import sys
import tarfile
import typing
import warnings
import weakref
import zipfile
from . import extra
from . import _extra
from . import utils
from .table import find_tables
class linkDest:
    """link or outline destination details"""

    def __init__(self, obj, rlink, document=None):
        isExt = obj.is_external
        isInt = not isExt
        self.dest = ''
        self.file_spec = ''
        self.flags = 0
        self.is_map = False
        self.is_uri = False
        self.kind = LINK_NONE
        self.lt = Point(0, 0)
        self.named = dict()
        self.new_window = ''
        self.page = obj.page
        self.rb = Point(0, 0)
        self.uri = obj.uri

        def uri_to_dict(uri):
            items = self.uri[1:].split('&')
            ret = dict()
            for item in items:
                eq = item.find('=')
                if eq >= 0:
                    ret[item[:eq]] = item[eq + 1:]
                else:
                    ret[item] = None
            return ret
        if rlink and (not self.uri.startswith('#')):
            self.uri = '#page=%i&zoom=0,%g,%g' % (rlink[0] + 1, rlink[1], rlink[2])
        if obj.is_external:
            self.page = -1
            self.kind = LINK_URI
        if not self.uri:
            self.page = -1
            self.kind = LINK_NONE
        if isInt and self.uri:
            self.uri = self.uri.replace('&zoom=nan', '&zoom=0')
            if self.uri.startswith('#'):
                self.kind = LINK_GOTO
                m = re.match('^#page=([0-9]+)&zoom=([0-9.]+),(-?[0-9.]+),(-?[0-9.]+)$', self.uri)
                if m:
                    self.page = int(m.group(1)) - 1
                    self.lt = Point(float(m.group(3)), float(m.group(4)))
                    self.flags = self.flags | LINK_FLAG_L_VALID | LINK_FLAG_T_VALID
                else:
                    m = re.match('^#page=([0-9]+)$', self.uri)
                    if m:
                        self.page = int(m.group(1)) - 1
                    else:
                        self.kind = LINK_NAMED
                        m = re.match('^#nameddest=(.*)', self.uri)
                        assert document
                        if document and m:
                            named = m.group(1)
                            self.named = document.resolve_names().get(named)
                            if self.named is None:
                                self.named = dict()
                            self.named['nameddest'] = named
                        else:
                            self.named = uri_to_dict(self.uri[1:])
            else:
                self.kind = LINK_NAMED
                self.named = uri_to_dict(self.uri)
        if obj.is_external:
            if not self.uri:
                pass
            elif self.uri.startswith('file:'):
                self.file_spec = self.uri[5:]
                if self.file_spec.startswith('//'):
                    self.file_spec = self.file_spec[2:]
                self.is_uri = False
                self.uri = ''
                self.kind = LINK_LAUNCH
                ftab = self.file_spec.split('#')
                if len(ftab) == 2:
                    if ftab[1].startswith('page='):
                        self.kind = LINK_GOTOR
                        self.file_spec = ftab[0]
                        self.page = int(ftab[1].split('&')[0][5:]) - 1
            elif ':' in self.uri:
                self.is_uri = True
                self.kind = LINK_URI
            else:
                self.is_uri = True
                self.kind = LINK_LAUNCH
        assert isinstance(self.named, dict)