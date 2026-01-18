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
def _from_cmdline(self, overrides):
    self._reset()
    for over in overrides:
        try:
            name, value = over.split('=', 1)
        except ValueError:
            raise errors.CommandError(gettext("Invalid '%s', should be of the form 'name=value'") % (over,))
        self.options[name] = value