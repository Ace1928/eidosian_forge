import logging
import os
import sys
import warnings
from traitlets.config.loader import Config
from traitlets.config.application import boolean_flag, catch_config_error
from IPython.core import release
from IPython.core import usage
from IPython.core.completer import IPCompleter
from IPython.core.crashhandler import CrashHandler
from IPython.core.formatters import PlainTextFormatter
from IPython.core.history import HistoryManager
from IPython.core.application import (
from IPython.core.magic import MagicsManager
from IPython.core.magics import (
from IPython.core.shellapp import (
from IPython.extensions.storemagic import StoreMagics
from .interactiveshell import TerminalInteractiveShell
from IPython.paths import get_ipython_dir
from traitlets import (
@default('classes')
def _classes_default(self):
    """This has to be in a method, for TerminalIPythonApp to be available."""
    return [InteractiveShellApp, self.__class__, TerminalInteractiveShell, HistoryManager, MagicsManager, ProfileDir, PlainTextFormatter, IPCompleter, ScriptMagics, LoggingMagics, StoreMagics]