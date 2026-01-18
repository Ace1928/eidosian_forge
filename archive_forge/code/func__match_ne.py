import logging
import re
from .compat import string_types
from .util import parse_requirement
def _match_ne(self, version, constraint, prefix):
    version, constraint = self._adjust_local(version, constraint, prefix)
    if not prefix:
        result = version != constraint
    else:
        result = not _match_prefix(version, constraint)
    return result