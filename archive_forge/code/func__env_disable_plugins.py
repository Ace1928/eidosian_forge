import os
import re
import sys
from importlib import util as importlib_util
import breezy
from . import debug, errors, osutils, trace
def _env_disable_plugins(key='BRZ_DISABLE_PLUGINS'):
    """Gives list of names for plugins to disable from environ key."""
    disabled_names = []
    env = os.environ.get(key)
    if env:
        for name in env.split(os.pathsep):
            name = _expect_identifier(name, key, env)
            if name is not None:
                disabled_names.append(name)
    return disabled_names