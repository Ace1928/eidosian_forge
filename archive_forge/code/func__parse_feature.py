from __future__ import print_function
import collections
import re
import sys
import codecs
from . import (
from .helpers import (
def _parse_feature(self, info):
    """Parse a feature command."""
    parts = info.split(b'=', 1)
    name = parts[0]
    if len(parts) > 1:
        value = self._path(parts[1])
    else:
        value = None
    self.features[name] = value
    return commands.FeatureCommand(name, value, lineno=self.lineno)