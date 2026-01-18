import os
import re
import sys
from importlib import util as importlib_util
import breezy
from . import debug, errors, osutils, trace
def _env_plugins_at(key='BRZ_PLUGINS_AT'):
    """Gives list of names and paths of specific plugins from environ key."""
    plugin_details = []
    env = os.environ.get(key)
    if env:
        for pair in env.split(os.pathsep):
            if '@' in pair:
                name, path = pair.split('@', 1)
            else:
                path = pair
                name = osutils.basename(path).split('.', 1)[0]
            name = _expect_identifier(name, key, env)
            if name is not None:
                plugin_details.append((name, os.path.abspath(path)))
    return plugin_details