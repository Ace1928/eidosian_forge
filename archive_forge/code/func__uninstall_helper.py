import collections
import os
import os.path
import subprocess
import sys
import sysconfig
import tempfile
from importlib import resources
import runpy
import sys
def _uninstall_helper(*, verbosity=0):
    """Helper to support a clean default uninstall process on Windows

    Note that calling this function may alter os.environ.
    """
    try:
        import pip
    except ImportError:
        return
    available_version = version()
    if pip.__version__ != available_version:
        print(f'ensurepip will only uninstall a matching version ({pip.__version__!r} installed, {available_version!r} available)', file=sys.stderr)
        return
    _disable_pip_configuration_settings()
    args = ['uninstall', '-y', '--disable-pip-version-check']
    if verbosity:
        args += ['-' + 'v' * verbosity]
    return _run_pip([*args, *reversed(_PACKAGE_NAMES)])