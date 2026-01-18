from __future__ import annotations
import re
from typing import Callable, Iterable, NamedTuple
from prompt_toolkit.document import Document
from prompt_toolkit.filters import FilterOrBool, to_filter
from prompt_toolkit.formatted_text import AnyFormattedText, StyleAndTextTuples
from .base import CompleteEvent, Completer, Completion
from .word_completer import WordCompleter
class _FuzzyMatch(NamedTuple):
    match_length: int
    start_pos: int
    completion: Completion