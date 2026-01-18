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
@classmethod
def _get_script_args(cls, type_, name, header, script_text):
    """
        For Windows, add a .py extension and an .exe launcher
        """
    if type_ == 'gui':
        launcher_type = 'gui'
        ext = '-script.pyw'
        old = ['.pyw']
    else:
        launcher_type = 'cli'
        ext = '-script.py'
        old = ['.py', '.pyc', '.pyo']
    hdr = cls._adjust_header(type_, header)
    blockers = [name + x for x in old]
    yield (name + ext, hdr + script_text, 't', blockers)
    yield (name + '.exe', get_win_launcher(launcher_type), 'b')
    if not is_64bit():
        m_name = name + '.exe.manifest'
        yield (m_name, load_launcher_manifest(name), 't')