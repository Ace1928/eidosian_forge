import functools
import os
import re
import _distutils_hack.override  # noqa: F401
import distutils.core
from distutils.errors import DistutilsOptionError
from distutils.util import convert_path as _convert_path
from . import logging, monkey
from . import version as _version_module
from .depends import Require
from .discovery import PackageFinder, PEP420PackageFinder
from .dist import Distribution
from .extension import Extension
from .warnings import SetuptoolsDeprecationWarning
class sic(str):
    """Treat this string as-is (https://en.wikipedia.org/wiki/Sic)"""