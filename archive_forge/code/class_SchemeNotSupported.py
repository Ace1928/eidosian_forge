from collections.abc import Sequence
from typing import Optional, Union, cast
from twisted.python.compat import nativeString
from twisted.web._responses import RESPONSES
class SchemeNotSupported(Exception):
    """
    The scheme of a URI was not one of the supported values.
    """