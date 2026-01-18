from __future__ import annotations
import logging # isort:skip
from math import inf
from typing import Any as any
from ...core.has_props import abstract
from ...core.properties import (
from ...util.deprecation import deprecated
from ..dom import HTML
from ..formatters import TickFormatter
from ..ui import Tooltip
from .widget import Widget
class AutocompleteInput(TextInput):
    """ Single-line input widget with auto-completion.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    completions = List(String, help='\n    A list of completion strings. This will be used to guide the\n    user upon typing the beginning of a desired value.\n    ')
    max_completions = Nullable(Positive(Int), help='\n    Optional maximum number of completions displayed.\n    ')
    min_characters = NonNegative(Int, default=2, help='\n    The number of characters a user must type before completions are presented.\n    ')
    case_sensitive = Bool(default=True, help='\n    Enable or disable case sensitivity.\n    ')
    restrict = Bool(default=True, help='\n    Set to False in order to allow users to enter text that is not present in the list of completion strings.\n    ')
    search_strategy = Enum('starts_with', 'includes', help='\n    Define how to search the list of completion strings. The default option\n    ``"starts_with"`` means that the user\'s text must match the start of a\n    completion string. Using ``"includes"`` means that the user\'s text can\n    match any substring of a completion string.\n    ')