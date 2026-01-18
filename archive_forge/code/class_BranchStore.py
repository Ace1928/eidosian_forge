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
class BranchStore(TransportIniFileStore):
    """A config store for branch options.

    There is a single BranchStore for a given branch.
    """

    def __init__(self, branch):
        super().__init__(branch.control_transport, 'branch.conf')
        self.branch = branch
        self.id = 'branch'