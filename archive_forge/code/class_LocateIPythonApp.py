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
class LocateIPythonApp(BaseIPythonApplication):
    description = 'print the path to the IPython dir'
    subcommands = dict(profile=('IPython.core.profileapp.ProfileLocate', 'print the path to an IPython profile directory'))

    def start(self):
        if self.subapp is not None:
            return self.subapp.start()
        else:
            print(self.ipython_dir)