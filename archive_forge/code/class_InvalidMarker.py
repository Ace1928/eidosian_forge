import operator
import os
import platform
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from ._parser import (
from ._parser import (
from ._tokenizer import ParserSyntaxError
from .specifiers import InvalidSpecifier, Specifier
from .utils import canonicalize_name
class InvalidMarker(ValueError):
    """
    An invalid marker was found, users should refer to PEP 508.
    """