import os
import platform
from copy import copy
from string import ascii_letters, digits
from typing import NamedTuple, Optional
from botocore import __version__ as botocore_version
from botocore.compat import HAS_CRT
class RawStringUserAgentComponent:
    """
    UserAgentComponent interface wrapper around ``str``.

    Use for User-Agent header components that are not constructed from
    prefix+name+value but instead are provided as strings. No sanitization is
    performed.
    """

    def __init__(self, value):
        self._value = value

    def to_string(self):
        return self._value