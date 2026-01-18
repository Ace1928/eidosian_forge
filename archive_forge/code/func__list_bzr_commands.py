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
def _list_bzr_commands(names):
    """Find commands from bzr's core and plugins.

    This is not the public interface, just the default hook called by
    all_command_names.
    """
    names.update(builtin_command_names())
    names.update(plugin_command_names())
    return names