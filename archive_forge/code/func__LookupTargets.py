import gyp.common
import json
import os
import posixpath
def _LookupTargets(names, mapping):
    """Returns a list of the mapping[name] for each value in |names| that is in
  |mapping|."""
    return [mapping[name] for name in names if name in mapping]