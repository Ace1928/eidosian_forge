import email
import itertools
import functools
import os
import posixpath
import re
import zipfile
import contextlib
from distutils.util import get_platform
import setuptools
from setuptools.extern.packaging.version import Version as parse_version
from setuptools.extern.packaging.tags import sys_tags
from setuptools.extern.packaging.utils import canonicalize_name
from setuptools.command.egg_info import write_requirements, _egg_basename
from setuptools.archive_util import _unpack_zipfile_obj
def _install_as_egg(self, destination_eggdir, zf):
    dist_basename = '%s-%s' % (self.project_name, self.version)
    dist_info = self.get_dist_info(zf)
    dist_data = '%s.data' % dist_basename
    egg_info = os.path.join(destination_eggdir, 'EGG-INFO')
    self._convert_metadata(zf, destination_eggdir, dist_info, egg_info)
    self._move_data_entries(destination_eggdir, dist_data)
    self._fix_namespace_packages(egg_info, destination_eggdir)