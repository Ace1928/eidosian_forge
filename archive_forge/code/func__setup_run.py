import contextlib
import os
import sys
from typing import List, Optional, Type, Union
from . import i18n, option, osutils, trace
from .lazy_import import lazy_import
import breezy
from breezy import (
from . import errors, registry
from .hooks import Hooks
from .i18n import gettext
from .plugin import disable_plugins, load_plugins, plugin_name
def _setup_run(self):
    """Wrap the defined run method on self with a cleanup.

        This is called by __init__ to make the Command be able to be run
        by just calling run(), as it could be before cleanups were added.

        If a different form of cleanups are in use by your Command subclass,
        you can override this method.
        """
    class_run = self.run

    def run(*args, **kwargs):
        for hook in Command.hooks['pre_command']:
            hook(self)
        try:
            with contextlib.ExitStack() as self._exit_stack:
                return class_run(*args, **kwargs)
        finally:
            for hook in Command.hooks['post_command']:
                hook(self)
    self.run = run