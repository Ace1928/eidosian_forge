import logging
import re
from .compat import string_types
from .util import parse_requirement
class SemanticVersion(Version):

    def parse(self, s):
        return _semantic_key(s)

    @property
    def is_prerelease(self):
        return self._parts[1][0] != '|'