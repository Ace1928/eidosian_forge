import gyp.common
import json
import os
import posixpath
def _NamesNotIn(names, mapping):
    """Returns a list of the values in |names| that are not in |mapping|."""
    return [name for name in names if name not in mapping]