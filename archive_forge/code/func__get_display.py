from __future__ import annotations
import re
from typing import Callable, Iterable, NamedTuple
from prompt_toolkit.document import Document
from prompt_toolkit.filters import FilterOrBool, to_filter
from prompt_toolkit.formatted_text import AnyFormattedText, StyleAndTextTuples
from .base import CompleteEvent, Completer, Completion
from .word_completer import WordCompleter
def _get_display(self, fuzzy_match: _FuzzyMatch, word_before_cursor: str) -> AnyFormattedText:
    """
        Generate formatted text for the display label.
        """

    def get_display() -> AnyFormattedText:
        m = fuzzy_match
        word = m.completion.text
        if m.match_length == 0:
            return m.completion.display
        result: StyleAndTextTuples = []
        result.append(('class:fuzzymatch.outside', word[:m.start_pos]))
        characters = list(word_before_cursor)
        for c in word[m.start_pos:m.start_pos + m.match_length]:
            classname = 'class:fuzzymatch.inside'
            if characters and c.lower() == characters[0].lower():
                classname += '.character'
                del characters[0]
            result.append((classname, c))
        result.append(('class:fuzzymatch.outside', word[m.start_pos + m.match_length:]))
        return result
    return get_display()