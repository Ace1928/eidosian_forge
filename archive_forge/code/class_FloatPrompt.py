from typing import Any, Generic, List, Optional, TextIO, TypeVar, Union, overload
from . import get_console
from .console import Console
from .text import Text, TextType
class FloatPrompt(PromptBase[int]):
    """A prompt that returns a float.

    Example:
        >>> temperature = FloatPrompt.ask("Enter desired temperature")

    """
    response_type = float
    validate_error_message = '[prompt.invalid]Please enter a number'