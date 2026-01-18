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
class LocationStore(LockableIniFileStore):
    """A config store for options specific to a location.

    There is a single LocationStore for a given process.
    """

    def __init__(self, possible_transports=None):
        t = transport.get_transport_from_path(bedding.config_dir(), possible_transports=possible_transports)
        super().__init__(t, 'locations.conf')
        self.id = 'locations'