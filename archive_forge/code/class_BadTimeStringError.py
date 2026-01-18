from __future__ import annotations
import math
from datetime import date, timedelta
from typing import Literal, overload
from streamlit.errors import MarkdownFormattedException, StreamlitAPIException
class BadTimeStringError(StreamlitAPIException):
    """Raised when a bad time string argument is passed."""

    def __init__(self, t: str):
        MarkdownFormattedException.__init__(self, f"Time string doesn't look right. It should be formatted as`'1d2h34m'` or `2 days`, for example. Got: {t}")