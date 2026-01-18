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
def __get_stubs_outputs(self):
    ns_ext_bases = (os.path.join(self.build_lib, *ext._full_name.split('.')) for ext in self.extensions if ext._needs_stub)
    pairs = itertools.product(ns_ext_bases, self.__get_output_extensions())
    return list((base + fnext for base, fnext in pairs))