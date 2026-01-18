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
def _get_cmd_object(cmd_name: str, plugins_override: bool=True, check_missing: bool=True) -> 'Command':
    """Get a command object.

    Args:
      cmd_name: The name of the command.
      plugins_override: Allow plugins to override builtins.
      check_missing: Look up commands not found in the regular index via
        the get_missing_command hook.

    Returns:
      A Command object instance

    Raises:
      KeyError: If no command is found.
    """
    cmd: Optional[Command] = None
    for hook in Command.hooks['get_command']:
        cmd = hook(cmd, cmd_name)
        if cmd is not None and (not plugins_override) and (not cmd.plugin_name()):
            break
    if cmd is None and check_missing:
        for hook in Command.hooks['get_missing_command']:
            cmd = hook(cmd_name)
            if cmd is not None:
                break
    if cmd is None:
        raise KeyError
    for hook in Command.hooks['extend_command']:
        hook(cmd)
    if getattr(cmd, 'invoked_as', None) is None:
        cmd.invoked_as = cmd_name
    return cmd