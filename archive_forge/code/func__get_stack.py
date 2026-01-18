import os
import sys
from io import BytesIO
from typing import Callable, Dict, Iterable, Tuple, cast
import configobj
import breezy
from .lazy_import import lazy_import
import errno
import fnmatch
import re
from breezy import (
from breezy.i18n import gettext
from . import (bedding, commands, errors, hooks, lazy_regex, registry, trace,
from .option import Option as CommandOption
def _get_stack(self, directory, scope=None, write_access=False):
    """Get the configuration stack specified by ``directory`` and ``scope``.

        Args:
          directory: Where the configurations are derived from.
          scope: A specific config to start from.
          write_access: Whether a write access to the stack will be
            attempted.
        """
    if scope is not None:
        if scope == 'breezy':
            return GlobalStack()
        elif scope == 'locations':
            return LocationStack(directory)
        elif scope == 'branch':
            _, br, _ = controldir.ControlDir.open_containing_tree_or_branch(directory)
            if write_access:
                self.add_cleanup(br.lock_write().unlock)
            return br.get_config_stack()
        raise NoSuchConfig(scope)
    else:
        try:
            _, br, _ = controldir.ControlDir.open_containing_tree_or_branch(directory)
            if write_access:
                self.add_cleanup(br.lock_write().unlock)
            return br.get_config_stack()
        except errors.NotBranchError:
            return LocationStack(directory)