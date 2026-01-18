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
class _CompatibleStack(Stack):
    """Place holder for compatibility with previous design.

    This is intended to ease the transition from the Config-based design to the
    Stack-based design and should not be used nor relied upon by plugins.

    One assumption made here is that the daughter classes will all use Stores
    derived from LockableIniFileStore).

    It implements set() and remove () by re-loading the store before applying
    the modification and saving it.

    The long term plan being to implement a single write by store to save
    all modifications, this class should not be used in the interim.
    """

    def set(self, name, value):
        self.store.unload()
        super().set(name, value)
        self.store.save()

    def remove(self, name):
        self.store.unload()
        super().remove(name)
        self.store.save()