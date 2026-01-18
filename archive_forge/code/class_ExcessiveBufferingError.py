from collections.abc import Sequence
from typing import Optional, Union, cast
from twisted.python.compat import nativeString
from twisted.web._responses import RESPONSES
class ExcessiveBufferingError(Exception):
    """
    The HTTP/2 protocol has been forced to buffer an excessive amount of
    outbound data, and has therefore closed the connection and dropped all
    outbound data.
    """