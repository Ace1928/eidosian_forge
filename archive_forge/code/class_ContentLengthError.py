from textwrap import indent
from typing import Optional, Union
from .typedefs import _CIMultiDict
class ContentLengthError(PayloadEncodingError):
    """Not enough data for satisfy content length header."""