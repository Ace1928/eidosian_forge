import re
from .api import FancyValidator
from .compound import Any
from .validators import Regex, Invalid, _
def _prepend_country_code(self, value, transformations, country_code):
    for rex, trf in transformations:
        match = rex.search(value)
        if match:
            return trf % ((country_code,) + match.groups())
    return value