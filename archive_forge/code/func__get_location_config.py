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
def _get_location_config(self):
    if self._location_config is None:
        if self.branch.base is None:
            self.branch.base = 'memory://'
        self._location_config = LocationConfig(self.branch.base)
    return self._location_config