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
def _get_user_option(self, option_name):
    """See Config._get_user_option."""
    for source in self.option_sources:
        value = source()._get_user_option(option_name)
        if value is not None:
            return value
    return None