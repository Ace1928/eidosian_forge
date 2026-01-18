import logging
import os
import sys
from distutils.cmd import Command as DistutilsCommand
from distutils.command.install import SCHEME_KEYS
from distutils.command.install import install as distutils_install_command
from distutils.sysconfig import get_python_lib
from typing import Dict, List, Optional, Union, cast
from pip._internal.models.scheme import Scheme
from pip._internal.utils.compat import WINDOWS
from pip._internal.utils.virtualenv import running_under_virtualenv
from .base import get_major_minor_version

    Get the "scheme" corresponding to the input parameters. The distutils
    documentation provides the context for the available schemes:
    https://docs.python.org/3/install/index.html#alternate-installation

    :param dist_name: the name of the package to retrieve the scheme for, used
        in the headers scheme path
    :param user: indicates to use the "user" scheme
    :param home: indicates to use the "home" scheme and provides the base
        directory for the same
    :param root: root under which other directories are re-based
    :param isolated: equivalent to --no-user-cfg, i.e. do not consider
        ~/.pydistutils.cfg (posix) or ~/pydistutils.cfg (non-posix) for
        scheme paths
    :param prefix: indicates to use the "prefix" scheme and provides the
        base directory for the same
    