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
def _fetch_build_eggs(dist):
    try:
        dist.fetch_build_eggs(dist.setup_requires)
    except Exception as ex:
        msg = "\n        It is possible a package already installed in your system\n        contains an version that is invalid according to PEP 440.\n        You can try `pip install --use-pep517` as a workaround for this problem,\n        or rely on a new virtual environment.\n\n        If the problem refers to a package that is not installed yet,\n        please contact that package's maintainers or distributors.\n        "
        if 'InvalidVersion' in ex.__class__.__name__:
            if hasattr(ex, 'add_note'):
                ex.add_note(msg)
            else:
                dist.announce(f'\n{msg}\n')
        raise