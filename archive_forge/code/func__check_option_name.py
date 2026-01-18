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
def _check_option_name(self, option_name):
    """Ensures an option name is valid.

        Args:
          option_name: The name to validate.
        """
    if _option_ref_re.match('{%s}' % option_name) is None:
        raise IllegalOptionName(option_name)