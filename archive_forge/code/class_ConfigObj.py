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
class ConfigObj(configobj.ConfigObj):

    def __init__(self, infile=None, **kwargs):
        super().__init__(infile=infile, interpolation=False, **kwargs)
    if _has_triplequote_bug():

        def _get_triple_quote(self, value):
            quot = super()._get_triple_quote(value)
            if quot == configobj.tdquot:
                return configobj.tsquot
            return configobj.tdquot

    def get_bool(self, section, key) -> bool:
        return cast(bool, self[section].as_bool(key))

    def get_value(self, section, name):
        if section == 'DEFAULT':
            try:
                return self[name]
            except KeyError:
                pass
        return self[section][name]