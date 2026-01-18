from collections.abc import Sequence
from typing import Optional, Union, cast
from twisted.python.compat import nativeString
from twisted.web._responses import RESPONSES
class RenderError(Exception):
    """
    Base exception class for all errors which can occur during template
    rendering.
    """