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
class GlobalStore(LockableIniFileStore):
    """A config store for global options.

    There is a single GlobalStore for a given process.
    """

    def __init__(self, possible_transports=None):
        path, kind = bedding._config_dir()
        t = transport.get_transport_from_path(path, possible_transports=possible_transports)
        super().__init__(t, kind + '.conf')
        self.id = 'breezy'