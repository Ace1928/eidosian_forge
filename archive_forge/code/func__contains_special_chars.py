from __future__ import annotations
import re
import textwrap
from typing import TYPE_CHECKING, Any, Final, cast
from streamlit.errors import StreamlitAPIException
def _contains_special_chars(text: str) -> bool:
    """Check if a string contains any special chars.

    Special chars in that case are all chars that are not
    alphanumeric, underscore, hyphen or whitespace.
    """
    return re.match(_ALPHANUMERIC_CHAR_REGEX, text) is None if text else False