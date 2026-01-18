import os
import re
import sys
from importlib import util as importlib_util
import breezy
from . import debug, errors, osutils, trace
def _iter_possible_plugins(plugin_paths):
    """Generate names and paths of possible plugins from plugin_paths."""
    yield from getattr(plugin_paths, 'extra_details', ())
    for path in plugin_paths:
        if os.path.isfile(path):
            if path.endswith('.zip'):
                trace.mutter("Don't yet support loading plugins from zip.")
        else:
            yield from _walk_modules(path)