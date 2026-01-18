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
class DistInfoDistribution(Distribution):
    """
    Wrap an actual or potential sys.path entry
    w/metadata, .dist-info style.
    """
    PKG_INFO = 'METADATA'
    EQEQ = re.compile('([\\(,])\\s*(\\d.*?)\\s*([,\\)])')

    @property
    def _parsed_pkg_info(self):
        """Parse and cache metadata"""
        try:
            return self._pkg_info
        except AttributeError:
            metadata = self.get_metadata(self.PKG_INFO)
            self._pkg_info = email.parser.Parser().parsestr(metadata)
            return self._pkg_info

    @property
    def _dep_map(self):
        try:
            return self.__dep_map
        except AttributeError:
            self.__dep_map = self._compute_dependencies()
            return self.__dep_map

    def _compute_dependencies(self):
        """Recompute this distribution's dependencies."""
        dm = self.__dep_map = {None: []}
        reqs = []
        for req in self._parsed_pkg_info.get_all('Requires-Dist') or []:
            reqs.extend(parse_requirements(req))

        def reqs_for_extra(extra):
            for req in reqs:
                if not req.marker or req.marker.evaluate({'extra': extra}):
                    yield req
        common = types.MappingProxyType(dict.fromkeys(reqs_for_extra(None)))
        dm[None].extend(common)
        for extra in self._parsed_pkg_info.get_all('Provides-Extra') or []:
            s_extra = safe_extra(extra.strip())
            dm[s_extra] = [r for r in reqs_for_extra(extra) if r not in common]
        return dm