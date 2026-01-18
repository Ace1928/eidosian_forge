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
def _get_triple_quote(self, value):
    quot = super()._get_triple_quote(value)
    if quot == configobj.tdquot:
        return configobj.tsquot
    return configobj.tdquot