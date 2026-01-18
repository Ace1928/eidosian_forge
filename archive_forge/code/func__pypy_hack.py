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
def _pypy_hack(name):
    PY37 = sys.version_info < (3, 8)
    old_pypy = hasattr(sys, 'pypy_version_info') and PY37
    prefix = not name.endswith(('_user', '_home'))
    pypy_name = 'pypy' + '_nt' * (os.name == 'nt')
    return pypy_name if old_pypy and prefix else name