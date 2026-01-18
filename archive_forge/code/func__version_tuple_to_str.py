import dataclasses
import re
from dataclasses import dataclass
from functools import total_ordering
from typing import Optional, Union
def _version_tuple_to_str(version_tuple):
    """Return the str version from the version tuple (major, minor, patch)."""
    return '.'.join((str(v) for v in version_tuple))