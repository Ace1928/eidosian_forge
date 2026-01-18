import sys
import warnings
from IPython.core import ultratb, compilerop
from IPython.core import magic_arguments
from IPython.core.magic import Magics, magics_class, line_magic
from IPython.core.interactiveshell import DummyMod, InteractiveShell
from IPython.terminal.interactiveshell import TerminalInteractiveShell
from IPython.terminal.ipapp import load_default_config
from traitlets import Bool, CBool, Unicode
from IPython.utils.io import ask_yes_no
from typing import Set
def _disable_init_location(self):
    """Disable the current Instance creation location"""
    InteractiveShellEmbed._inactive_locations.add(self._init_location_id)