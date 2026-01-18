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
def add_cleanup(self, cleanup_func, *args, **kwargs):
    """Register a function to call after self.run returns or raises.

        Functions will be called in LIFO order.
        """
    self._exit_stack.callback(cleanup_func, *args, **kwargs)