from __future__ import division, print_function
import sys
import os
import io
import re
import glob
import math
import zlib
import time
import json
import enum
import struct
import pathlib
import warnings
import binascii
import tempfile
import datetime
import threading
import collections
import multiprocessing
import concurrent.futures
import numpy
class TiffPages(object):
    """Sequence of TIFF image file directories."""

    def __init__(self, parent):
        """Initialize instance from file. Read first TiffPage from file.

        The file position must be at an offset to an offset to a TiffPage.

        """
        self.parent = parent
        self.pages = []
        self.complete = False
        self._tiffpage = TiffPage
        self._keyframe = None
        self._cache = True
        fh = parent.filehandle
        self._nextpageoffset = fh.tell()
        offset = struct.unpack(parent.offsetformat, fh.read(parent.offsetsize))[0]
        if offset == 0:
            self.complete = True
            return
        if offset >= fh.size:
            warnings.warn('invalid page offset (%i)' % offset)
            self.complete = True
            return
        fh.seek(offset)
        page = TiffPage(parent, index=0)
        self.pages.append(page)
        self._keyframe = page

    @property
    def cache(self):
        """Return if pages/frames are currently being cached."""
        return self._cache

    @cache.setter
    def cache(self, value):
        """Enable or disable caching of pages/frames. Clear cache if False."""
        value = bool(value)
        if self._cache and (not value):
            self.clear()
        self._cache = value

    @property
    def useframes(self):
        """Return if currently using TiffFrame (True) or TiffPage (False)."""
        return self._tiffpage == TiffFrame and TiffFrame is not TiffPage

    @useframes.setter
    def useframes(self, value):
        """Set to use TiffFrame (True) or TiffPage (False)."""
        self._tiffpage = TiffFrame if value else TiffPage

    @property
    def keyframe(self):
        """Return index of current keyframe."""
        return self._keyframe.index

    @keyframe.setter
    def keyframe(self, index):
        """Set current keyframe. Load TiffPage from file if necessary."""
        if self._keyframe.index == index:
            return
        if self.complete or 0 <= index < len(self.pages):
            page = self.pages[index]
            if isinstance(page, TiffPage):
                self._keyframe = page
                return
            elif isinstance(page, TiffFrame):
                self.pages[index] = page.offset
        useframes = self.useframes
        self._tiffpage = TiffPage
        self._keyframe = self[index]
        self.useframes = useframes

    @property
    def next_page_offset(self):
        """Return offset where offset to a new page can be stored."""
        if not self.complete:
            self._seek(-1)
        return self._nextpageoffset

    def load(self):
        """Read all remaining pages from file."""
        fh = self.parent.filehandle
        keyframe = self._keyframe
        pages = self.pages
        if not self.complete:
            self._seek(-1)
        for i, page in enumerate(pages):
            if isinstance(page, inttypes):
                fh.seek(page)
                page = self._tiffpage(self.parent, index=i, keyframe=keyframe)
                pages[i] = page

    def clear(self, fully=True):
        """Delete all but first page from cache. Set keyframe to first page."""
        pages = self.pages
        if not self._cache or len(pages) < 1:
            return
        self._keyframe = pages[0]
        if fully:
            for i, page in enumerate(pages[1:]):
                if not isinstance(page, inttypes):
                    pages[i + 1] = page.offset
        elif TiffFrame is not TiffPage:
            for i, page in enumerate(pages):
                if isinstance(page, TiffFrame):
                    pages[i] = page.offset

    def _seek(self, index, maxpages=2 ** 22):
        """Seek file to offset of specified page."""
        pages = self.pages
        if not pages:
            return
        fh = self.parent.filehandle
        if fh.closed:
            raise RuntimeError('FileHandle is closed')
        if self.complete or 0 <= index < len(pages):
            page = pages[index]
            offset = page if isinstance(page, inttypes) else page.offset
            fh.seek(offset)
            return
        offsetformat = self.parent.offsetformat
        offsetsize = self.parent.offsetsize
        tagnoformat = self.parent.tagnoformat
        tagnosize = self.parent.tagnosize
        tagsize = self.parent.tagsize
        unpack = struct.unpack
        page = pages[-1]
        offset = page if isinstance(page, inttypes) else page.offset
        while len(pages) < maxpages:
            fh.seek(offset)
            try:
                tagno = unpack(tagnoformat, fh.read(tagnosize))[0]
                if tagno > 4096:
                    raise ValueError('suspicious number of tags')
            except Exception:
                warnings.warn('corrupted tag list at offset %i' % offset)
                del pages[-1]
                self.complete = True
                break
            self._nextpageoffset = offset + tagnosize + tagno * tagsize
            fh.seek(self._nextpageoffset)
            offset = unpack(offsetformat, fh.read(offsetsize))[0]
            if offset == 0:
                self.complete = True
                break
            if offset >= fh.size:
                warnings.warn('invalid page offset (%i)' % offset)
                self.complete = True
                break
            pages.append(offset)
            if 0 <= index < len(pages):
                break
        if index >= len(pages):
            raise IndexError('list index out of range')
        page = pages[index]
        fh.seek(page if isinstance(page, inttypes) else page.offset)

    def __bool__(self):
        """Return True if file contains any pages."""
        return len(self.pages) > 0

    def __len__(self):
        """Return number of pages in file."""
        if not self.complete:
            self._seek(-1)
        return len(self.pages)

    def __getitem__(self, key):
        """Return specified page(s) from cache or file."""
        pages = self.pages
        if not pages:
            raise IndexError('list index out of range')
        if key == 0:
            return pages[key]
        if isinstance(key, slice):
            start, stop, _ = key.indices(2 ** 31 - 1)
            if not self.complete and max(stop, start) > len(pages):
                self._seek(-1)
            return [self[i] for i in range(*key.indices(len(pages)))]
        if self.complete and key >= len(pages):
            raise IndexError('list index out of range')
        try:
            page = pages[key]
        except IndexError:
            page = 0
        if not isinstance(page, inttypes):
            return page
        self._seek(key)
        page = self._tiffpage(self.parent, index=key, keyframe=self._keyframe)
        if self._cache:
            pages[key] = page
        return page

    def __iter__(self):
        """Return iterator over all pages."""
        i = 0
        while True:
            try:
                yield self[i]
                i += 1
            except IndexError:
                break