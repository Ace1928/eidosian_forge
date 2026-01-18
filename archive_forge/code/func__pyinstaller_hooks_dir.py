import sys
import warnings
from ._globals import _NoValue, _CopyMode
from .exceptions import (
from . import version
from .version import __version__
def _pyinstaller_hooks_dir():
    from pathlib import Path
    return [str(Path(__file__).with_name('_pyinstaller').resolve())]