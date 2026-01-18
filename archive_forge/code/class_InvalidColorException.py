from __future__ import annotations
from typing import Any, Callable, Collection, Tuple, Union, cast
from typing_extensions import TypeAlias
from streamlit.errors import StreamlitAPIException
class InvalidColorException(StreamlitAPIException):

    def __init__(self, color, *args):
        message = f"This does not look like a valid color: {repr(color)}.\n\nColors must be in one of the following formats:\n\n* Hex string with 3, 4, 6, or 8 digits. Example: `'#00ff00'`\n* List or tuple with 3 or 4 components. Example: `[1.0, 0.5, 0, 0.2]`\n            "
        super().__init__(message, *args)