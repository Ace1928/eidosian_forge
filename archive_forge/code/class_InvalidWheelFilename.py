import re
from typing import FrozenSet, NewType, Tuple, Union, cast
from .tags import Tag, parse_tag
from .version import InvalidVersion, Version
class InvalidWheelFilename(ValueError):
    """
    An invalid wheel filename was found, users should refer to PEP 427.
    """