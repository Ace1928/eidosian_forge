import os
import re
import sys
from importlib import util as importlib_util
import breezy
from . import debug, errors, osutils, trace
def _iter_plugin_paths(paths_from_env, core_paths):
    """Generate paths using paths_from_env and core_paths."""
    for path, context in paths_from_env:
        if context == 'path':
            yield os.path.abspath(path)
        elif context == 'user':
            path = get_user_plugin_path()
            if os.path.isdir(path):
                yield path
        elif context == 'core':
            for path in _get_core_plugin_paths(core_paths):
                yield path
        elif context == 'site':
            for path in _get_site_plugin_paths(sys.path):
                if os.path.isdir(path):
                    yield path