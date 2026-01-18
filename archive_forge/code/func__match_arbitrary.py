import logging
import re
from .compat import string_types
from .util import parse_requirement
def _match_arbitrary(self, version, constraint, prefix):
    return str(version) == str(constraint)