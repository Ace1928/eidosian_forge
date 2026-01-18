from distutils.filelist import FileList as _FileList
from distutils.errors import DistutilsInternalError
from distutils.util import convert_path
from distutils import log
import distutils.errors
import distutils.filelist
import functools
import os
import re
import sys
import time
import collections
from .._importlib import metadata
from .. import _entry_points, _normalization
from . import _requirestxt
from setuptools import Command
from setuptools.command.sdist import sdist
from setuptools.command.sdist import walk_revctrl
from setuptools.command.setopt import edit_config
from setuptools.command import bdist_egg
import setuptools.unicode_utils as unicode_utils
from setuptools.glob import glob
from setuptools.extern import packaging
from ..warnings import SetuptoolsDeprecationWarning
class InfoCommon:
    tag_build = None
    tag_date = None

    @property
    def name(self):
        return _normalization.safe_name(self.distribution.get_name())

    def tagged_version(self):
        tagged = self._maybe_tag(self.distribution.get_version())
        return _normalization.best_effort_version(tagged)

    def _maybe_tag(self, version):
        """
        egg_info may be called more than once for a distribution,
        in which case the version string already contains all tags.
        """
        return version if self.vtags and self._already_tagged(version) else version + self.vtags

    def _already_tagged(self, version: str) -> bool:
        return version.endswith(self.vtags) or version.endswith(self._safe_tags())

    def _safe_tags(self) -> str:
        return _normalization.best_effort_version(f'0{self.vtags}')[1:]

    def tags(self) -> str:
        version = ''
        if self.tag_build:
            version += self.tag_build
        if self.tag_date:
            version += time.strftime('%Y%m%d')
        return version
    vtags = property(tags)