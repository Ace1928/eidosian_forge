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
def _set_configobj(self, configobj):
    out_file = BytesIO()
    configobj.write(out_file)
    out_file.seek(0)
    self._transport.put_file(self._filename, out_file)
    for hook in OldConfigHooks['save']:
        hook(self)