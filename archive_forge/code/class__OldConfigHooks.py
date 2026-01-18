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
class _OldConfigHooks(hooks.Hooks):
    """A dict mapping hook names and a list of callables for configs.
    """

    def __init__(self):
        """Create the default hooks.

        These are all empty initially, because by default nothing should get
        notified.
        """
        super().__init__('breezy.config', 'OldConfigHooks')
        self.add_hook('load', 'Invoked when a config store is loaded. The signature is (config).', (2, 4))
        self.add_hook('save', 'Invoked when a config store is saved. The signature is (config).', (2, 4))
        self.add_hook('get', 'Invoked when a config option is read. The signature is (config, name, value).', (2, 4))
        self.add_hook('set', 'Invoked when a config option is set. The signature is (config, name, value).', (2, 4))
        self.add_hook('remove', 'Invoked when a config option is removed. The signature is (config, name).', (2, 4))