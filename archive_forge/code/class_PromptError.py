from typing import Any, Generic, List, Optional, TextIO, TypeVar, Union, overload
from . import get_console
from .console import Console
from .text import Text, TextType
class PromptError(Exception):
    """Exception base class for prompt related errors."""