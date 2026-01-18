import re
from .api import FancyValidator
from .compound import Any
from .validators import Regex, Invalid, _
def _perform_rex_transformation(self, value, transformations):
    for rex, trf in transformations:
        match = rex.search(value)
        if match:
            value = trf % match.groups()
    return value