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
def _scan_module_for_commands(module):
    module_dict = module.__dict__
    for name in module_dict:
        if name.startswith('cmd_'):
            yield module_dict[name]