import logging
import re
from .compat import string_types
from .util import parse_requirement
class SemanticMatcher(Matcher):
    version_class = SemanticVersion