import os
import sys
import itertools
from importlib.machinery import EXTENSION_SUFFIXES
from importlib.util import cache_from_source as _compiled_file_name
from typing import Dict, Iterator, List, Tuple
from pathlib import Path
from distutils.command.build_ext import build_ext as _du_build_ext
from distutils.ccompiler import new_compiler
from distutils.sysconfig import customize_compiler, get_config_var
from distutils import log
from setuptools.errors import BaseError
from setuptools.extension import Extension, Library
from distutils.sysconfig import _config_vars as _CONFIG_VARS  # noqa
def _get_inplace_equivalent(self, build_py, ext: Extension) -> Tuple[str, str]:
    fullname = self.get_ext_fullname(ext.name)
    filename = self.get_ext_filename(fullname)
    modpath = fullname.split('.')
    package = '.'.join(modpath[:-1])
    package_dir = build_py.get_package_dir(package)
    inplace_file = os.path.join(package_dir, os.path.basename(filename))
    regular_file = os.path.join(self.build_lib, filename)
    return (inplace_file, regular_file)