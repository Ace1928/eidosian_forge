from hyperframe.exceptions import InvalidFrameError
from hyperframe.frame import (
from .exceptions import (

        Updates the internal header buffer. Returns a frame that should replace
        the current one. May throw exceptions if this frame is invalid.
        