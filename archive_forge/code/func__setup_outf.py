import contextlib
import os
import sys
from typing import List, Optional, Type, Union
from . import i18n, option, osutils, trace
from .lazy_import import lazy_import
import breezy
from breezy import (
from . import errors, registry
from .hooks import Hooks
from .i18n import gettext
from .plugin import disable_plugins, load_plugins, plugin_name
def _setup_outf(self):
    """Return a file linked to stdout, which has proper encoding."""
    self.outf = ui.ui_factory.make_output_stream(encoding_type=self.encoding_type)