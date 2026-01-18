import sys
import os
import contextlib
import sysconfig
import itertools
from distutils._log import log
from ..core import Command
from ..debug import DEBUG
from ..sysconfig import get_config_vars
from ..file_util import write_file
from ..util import convert_path, subst_vars, change_root
from ..util import get_platform
from ..errors import DistutilsOptionError, DistutilsPlatformError
from . import _framework_compat as fw
from .. import _collections
from site import USER_BASE
from site import USER_SITE
def _load_schemes():
    """
    Extend default schemes with schemes from sysconfig.
    """
    sysconfig_schemes = _load_sysconfig_schemes() or {}
    return {scheme: {**INSTALL_SCHEMES.get(scheme, {}), **sysconfig_schemes.get(scheme, {})} for scheme in set(itertools.chain(INSTALL_SCHEMES, sysconfig_schemes))}