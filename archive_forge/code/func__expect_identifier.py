import os
import re
import sys
from importlib import util as importlib_util
import breezy
from . import debug, errors, osutils, trace
def _expect_identifier(name, env_key, env_value):
    """Validate given name from envvar is usable as a Python identifier.

    Returns the name as a native str, or None if it was invalid.

    Per PEP 3131 this is no longer strictly correct for Python 3, but as MvL
    didn't include a neat way to check except eval, this enforces ascii.
    """
    if re.match('^[A-Za-z_][A-Za-z0-9_]*$', name) is None:
        trace.warning("Invalid name '%s' in %s='%s'", name, env_key, env_value)
        return None
    return str(name)