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
class MemoryStack(Stack):
    """A configuration stack defined from a string.

    This is mainly intended for tests and requires no disk resources.
    """

    def __init__(self, content=None):
        """Create an in-memory stack from a given content.

        It uses a single store based on configobj and support reading and
        writing options.

        Args:
          content: The initial content of the store. If None, the store is
            not loaded and ``_load_from_string`` can and should be used if
            needed.
        """
        store = IniFileStore()
        if content is not None:
            store._load_from_string(content)
        super().__init__([store.get_sections], store)