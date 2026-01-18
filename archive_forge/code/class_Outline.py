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
class Outline:

    def __init__(self, ol):
        self.this = ol

    @property
    def dest(self):
        """outline destination details"""
        return linkDest(self, None, None)

    def destination(self, document):
        """
        Like `dest` property but uses `document` to resolve destinations for
        kind=LINK_NAMED.
        """
        return linkDest(self, None, document)

    @property
    def down(self):
        ol = self.this
        down_ol = ol.down()
        if not down_ol.m_internal:
            return
        return Outline(down_ol)

    @property
    def is_external(self):
        if g_use_extra:
            return _extra.Outline_is_external(self.this)
        ol = self.this
        if not ol.m_internal:
            return False
        uri = ol.m_internal.uri if 1 else ol.uri()
        if uri is None:
            return False
        return mupdf.fz_is_external_link(uri)

    @property
    def is_open(self):
        if 1:
            return self.this.m_internal.is_open
        return self.this.is_open()

    @property
    def next(self):
        ol = self.this
        next_ol = ol.next()
        if not next_ol.m_internal:
            return
        return Outline(next_ol)

    @property
    def page(self):
        if 1:
            return self.this.m_internal.page.page
        return self.this.page().page

    @property
    def title(self):
        return self.this.m_internal.title

    @property
    def uri(self):
        ol = self.this
        if not ol.m_internal:
            return None
        return ol.m_internal.uri

    @property
    def x(self):
        return self.this.m_internal.x

    @property
    def y(self):
        return self.this.m_internal.y
    __slots__ = ['this']