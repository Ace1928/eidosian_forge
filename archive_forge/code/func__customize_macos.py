import functools
import os
import pathlib
import re
import sys
import sysconfig
from . import py39compat
from ._functools import pass_none
from .errors import DistutilsPlatformError
@functools.lru_cache
def _customize_macos():
    """
    Perform first-time customization of compiler-related
    config vars on macOS. Use after a compiler is known
    to be needed. This customization exists primarily to support Pythons
    from binary installers. The kind and paths to build tools on
    the user system may vary significantly from the system
    that Python itself was built on.  Also the user OS
    version and build tools may not support the same set
    of CPU architectures for universal builds.
    """
    sys.platform == 'darwin' and __import__('_osx_support').customize_compiler(get_config_vars())