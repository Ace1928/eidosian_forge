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
def _register_builtin_commands():
    if builtin_command_registry.keys():
        return
    import breezy.builtins
    for cmd_class in _scan_module_for_commands(breezy.builtins):
        builtin_command_registry.register(cmd_class)
    breezy.builtins._register_lazy_builtins()