import logging
import re
from .compat import string_types
from .util import parse_requirement
def _match_gt(self, version, constraint, prefix):
    version, constraint = self._adjust_local(version, constraint, prefix)
    if version <= constraint:
        return False
    release_clause = constraint._release_clause
    pfx = '.'.join([str(i) for i in release_clause])
    return not _match_prefix(version, pfx)