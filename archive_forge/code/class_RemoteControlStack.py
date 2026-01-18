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
class RemoteControlStack(Stack):
    """Remote control-only options stack."""

    def __init__(self, bzrdir):
        cstore = bzrdir._get_config_store()
        super().__init__([NameMatcher(cstore, None).get_sections], cstore)
        self.controldir = bzrdir