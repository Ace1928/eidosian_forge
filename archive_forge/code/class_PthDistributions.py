from glob import glob
from distutils.util import get_platform
from distutils.util import convert_path, subst_vars
from distutils.errors import (
from distutils import log, dir_util
from distutils.command.build_scripts import first_line_re
from distutils.spawn import find_executable
from distutils.command import install
import sys
import os
from typing import Dict, List
import zipimport
import shutil
import tempfile
import zipfile
import re
import stat
import random
import textwrap
import warnings
import site
import struct
import contextlib
import subprocess
import shlex
import io
import configparser
import sysconfig
from sysconfig import get_path
from setuptools import Command
from setuptools.sandbox import run_setup
from setuptools.command import setopt
from setuptools.archive_util import unpack_archive
from setuptools.package_index import (
from setuptools.command import bdist_egg, egg_info
from setuptools.warnings import SetuptoolsDeprecationWarning, SetuptoolsWarning
from setuptools.wheel import Wheel
from pkg_resources import (
import pkg_resources
from ..compat import py39, py311
from .._path import ensure_directory
from ..extern.jaraco.text import yield_lines
class PthDistributions(Environment):
    """A .pth file with Distribution paths in it"""

    def __init__(self, filename, sitedirs=()):
        self.filename = filename
        self.sitedirs = list(map(normalize_path, sitedirs))
        self.basedir = normalize_path(os.path.dirname(self.filename))
        self.paths, self.dirty = self._load()
        self._init_paths = self.paths[:]
        super().__init__([], None, None)
        for path in yield_lines(self.paths):
            list(map(self.add, find_distributions(path, True)))

    def _load_raw(self):
        paths = []
        dirty = saw_import = False
        seen = dict.fromkeys(self.sitedirs)
        f = open(self.filename, 'rt', encoding=py39.LOCALE_ENCODING)
        for line in f:
            path = line.rstrip()
            paths.append(path)
            if line.startswith(('import ', 'from ')):
                saw_import = True
                continue
            stripped_path = path.strip()
            if not stripped_path or stripped_path.startswith('#'):
                continue
            normalized_path = normalize_path(os.path.join(self.basedir, path))
            if normalized_path in seen or not os.path.exists(normalized_path):
                log.debug('cleaned up dirty or duplicated %r', path)
                dirty = True
                paths.pop()
                continue
            seen[normalized_path] = 1
        f.close()
        while paths and (not paths[-1].strip()):
            paths.pop()
            dirty = True
        return (paths, dirty or (paths and saw_import))

    def _load(self):
        if os.path.isfile(self.filename):
            return self._load_raw()
        return ([], False)

    def save(self):
        """Write changed .pth file back to disk"""
        last_paths, last_dirty = self._load()
        for path in last_paths[:]:
            if path not in self.paths:
                self.paths.append(path)
                log.info('detected new path %r', path)
                last_dirty = True
            else:
                last_paths.remove(path)
        for path in self.paths[:]:
            if path not in last_paths and (not path.startswith(('import ', 'from ', '#'))):
                absolute_path = os.path.join(self.basedir, path)
                if not os.path.exists(absolute_path):
                    self.paths.remove(path)
                    log.info('removing now non-existent path %r', path)
                    last_dirty = True
        self.dirty |= last_dirty or self.paths != self._init_paths
        if not self.dirty:
            return
        rel_paths = list(map(self.make_relative, self.paths))
        if rel_paths:
            log.debug('Saving %s', self.filename)
            lines = self._wrap_lines(rel_paths)
            data = '\n'.join(lines) + '\n'
            if os.path.islink(self.filename):
                os.unlink(self.filename)
            with open(self.filename, 'wt', encoding=py39.LOCALE_ENCODING) as f:
                f.write(data)
        elif os.path.exists(self.filename):
            log.debug('Deleting empty %s', self.filename)
            os.unlink(self.filename)
        self.dirty = False
        self._init_paths[:] = self.paths[:]

    @staticmethod
    def _wrap_lines(lines):
        return lines

    def add(self, dist):
        """Add `dist` to the distribution map"""
        new_path = dist.location not in self.paths and (dist.location not in self.sitedirs or dist.location == os.getcwd())
        if new_path:
            self.paths.append(dist.location)
            self.dirty = True
        super().add(dist)

    def remove(self, dist):
        """Remove `dist` from the distribution map"""
        while dist.location in self.paths:
            self.paths.remove(dist.location)
            self.dirty = True
        super().remove(dist)

    def make_relative(self, path):
        npath, last = os.path.split(normalize_path(path))
        baselen = len(self.basedir)
        parts = [last]
        sep = os.altsep == '/' and '/' or os.sep
        while len(npath) >= baselen:
            if npath == self.basedir:
                parts.append(os.curdir)
                parts.reverse()
                return sep.join(parts)
            npath, last = os.path.split(npath)
            parts.append(last)
        else:
            return path