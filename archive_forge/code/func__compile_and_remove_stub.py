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
def _compile_and_remove_stub(self, stub_file: str):
    from distutils.util import byte_compile
    byte_compile([stub_file], optimize=0, force=True, dry_run=self.dry_run)
    optimize = self.get_finalized_command('install_lib').optimize
    if optimize > 0:
        byte_compile([stub_file], optimize=optimize, force=True, dry_run=self.dry_run)
    if os.path.exists(stub_file) and (not self.dry_run):
        os.unlink(stub_file)