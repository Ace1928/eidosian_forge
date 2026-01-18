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
@staticmethod
def _fix_namespace_packages(egg_info, destination_eggdir):
    namespace_packages = os.path.join(egg_info, 'namespace_packages.txt')
    if os.path.exists(namespace_packages):
        with open(namespace_packages) as fp:
            namespace_packages = fp.read().split()
        for mod in namespace_packages:
            mod_dir = os.path.join(destination_eggdir, *mod.split('.'))
            mod_init = os.path.join(mod_dir, '__init__.py')
            if not os.path.exists(mod_dir):
                os.mkdir(mod_dir)
            if not os.path.exists(mod_init):
                with open(mod_init, 'w') as fp:
                    fp.write(NAMESPACE_PACKAGE_INIT)