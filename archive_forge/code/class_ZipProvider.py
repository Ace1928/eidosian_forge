import sys
import os
import io
import time
import re
import types
from typing import Protocol
import zipfile
import zipimport
import warnings
import stat
import functools
import pkgutil
import operator
import platform
import collections
import plistlib
import email.parser
import errno
import tempfile
import textwrap
import inspect
import ntpath
import posixpath
import importlib
import importlib.machinery
from pkgutil import get_importer
import _imp
from os import utime
from os import open as os_open
from os.path import isdir, split
from pkg_resources.extern.jaraco.text import (
from pkg_resources.extern import platformdirs
from pkg_resources.extern import packaging
class ZipProvider(EggProvider):
    """Resource support for zips and eggs"""
    eagers = None
    _zip_manifests = MemoizedZipManifests()

    def __init__(self, module):
        super().__init__(module)
        self.zip_pre = self.loader.archive + os.sep

    def _zipinfo_name(self, fspath):
        fspath = fspath.rstrip(os.sep)
        if fspath == self.loader.archive:
            return ''
        if fspath.startswith(self.zip_pre):
            return fspath[len(self.zip_pre):]
        raise AssertionError('%s is not a subpath of %s' % (fspath, self.zip_pre))

    def _parts(self, zip_path):
        fspath = self.zip_pre + zip_path
        if fspath.startswith(self.egg_root + os.sep):
            return fspath[len(self.egg_root) + 1:].split(os.sep)
        raise AssertionError('%s is not a subpath of %s' % (fspath, self.egg_root))

    @property
    def zipinfo(self):
        return self._zip_manifests.load(self.loader.archive)

    def get_resource_filename(self, manager, resource_name):
        if not self.egg_name:
            raise NotImplementedError('resource_filename() only supported for .egg, not .zip')
        zip_path = self._resource_to_zip(resource_name)
        eagers = self._get_eager_resources()
        if '/'.join(self._parts(zip_path)) in eagers:
            for name in eagers:
                self._extract_resource(manager, self._eager_to_zip(name))
        return self._extract_resource(manager, zip_path)

    @staticmethod
    def _get_date_and_size(zip_stat):
        size = zip_stat.file_size
        date_time = zip_stat.date_time + (0, 0, -1)
        timestamp = time.mktime(date_time)
        return (timestamp, size)

    def _extract_resource(self, manager, zip_path):
        if zip_path in self._index():
            for name in self._index()[zip_path]:
                last = self._extract_resource(manager, os.path.join(zip_path, name))
            return os.path.dirname(last)
        timestamp, size = self._get_date_and_size(self.zipinfo[zip_path])
        if not WRITE_SUPPORT:
            raise OSError('"os.rename" and "os.unlink" are not supported on this platform')
        try:
            real_path = manager.get_cache_path(self.egg_name, self._parts(zip_path))
            if self._is_current(real_path, zip_path):
                return real_path
            outf, tmpnam = _mkstemp('.$extract', dir=os.path.dirname(real_path))
            os.write(outf, self.loader.get_data(zip_path))
            os.close(outf)
            utime(tmpnam, (timestamp, timestamp))
            manager.postprocess(tmpnam, real_path)
            try:
                rename(tmpnam, real_path)
            except OSError:
                if os.path.isfile(real_path):
                    if self._is_current(real_path, zip_path):
                        return real_path
                    elif os.name == 'nt':
                        unlink(real_path)
                        rename(tmpnam, real_path)
                        return real_path
                raise
        except OSError:
            manager.extraction_error()
        return real_path

    def _is_current(self, file_path, zip_path):
        """
        Return True if the file_path is current for this zip_path
        """
        timestamp, size = self._get_date_and_size(self.zipinfo[zip_path])
        if not os.path.isfile(file_path):
            return False
        stat = os.stat(file_path)
        if stat.st_size != size or stat.st_mtime != timestamp:
            return False
        zip_contents = self.loader.get_data(zip_path)
        with open(file_path, 'rb') as f:
            file_contents = f.read()
        return zip_contents == file_contents

    def _get_eager_resources(self):
        if self.eagers is None:
            eagers = []
            for name in ('native_libs.txt', 'eager_resources.txt'):
                if self.has_metadata(name):
                    eagers.extend(self.get_metadata_lines(name))
            self.eagers = eagers
        return self.eagers

    def _index(self):
        try:
            return self._dirindex
        except AttributeError:
            ind = {}
            for path in self.zipinfo:
                parts = path.split(os.sep)
                while parts:
                    parent = os.sep.join(parts[:-1])
                    if parent in ind:
                        ind[parent].append(parts[-1])
                        break
                    else:
                        ind[parent] = [parts.pop()]
            self._dirindex = ind
            return ind

    def _has(self, fspath):
        zip_path = self._zipinfo_name(fspath)
        return zip_path in self.zipinfo or zip_path in self._index()

    def _isdir(self, fspath):
        return self._zipinfo_name(fspath) in self._index()

    def _listdir(self, fspath):
        return list(self._index().get(self._zipinfo_name(fspath), ()))

    def _eager_to_zip(self, resource_name):
        return self._zipinfo_name(self._fn(self.egg_root, resource_name))

    def _resource_to_zip(self, resource_name):
        return self._zipinfo_name(self._fn(self.module_path, resource_name))