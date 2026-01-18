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
def _get_safe_value(self, option_name):
    """This variant of get_best_value never returns untrusted values.

        It does not return values from the branch data, because the branch may
        not be controlled by the user.

        We may wish to allow locations.conf to control whether branches are
        trusted in the future.
        """
    for source in (self._get_location_config, self._get_global_config):
        value = getattr(source(), option_name)()
        if value is not None:
            return value
    return None