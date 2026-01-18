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
def _set_option_policy(self, section, option_name, option_policy):
    """Set the policy for the given option name in the given section."""
    policy_key = option_name + ':policy'
    policy_name = _policy_name[option_policy]
    if policy_name is not None:
        self._get_parser()[section][policy_key] = policy_name
    elif policy_key in self._get_parser()[section]:
        del self._get_parser()[section][policy_key]