from __future__ import annotations
import re
from typing import Callable, Iterable, NamedTuple
from prompt_toolkit.document import Document
from prompt_toolkit.filters import FilterOrBool, to_filter
from prompt_toolkit.formatted_text import AnyFormattedText, StyleAndTextTuples
from .base import CompleteEvent, Completer, Completion
from .word_completer import WordCompleter
class FuzzyWordCompleter(Completer):
    """
    Fuzzy completion on a list of words.

    (This is basically a `WordCompleter` wrapped in a `FuzzyCompleter`.)

    :param words: List of words or callable that returns a list of words.
    :param meta_dict: Optional dict mapping words to their meta-information.
    :param WORD: When True, use WORD characters.
    """

    def __init__(self, words: list[str] | Callable[[], list[str]], meta_dict: dict[str, str] | None=None, WORD: bool=False) -> None:
        self.words = words
        self.meta_dict = meta_dict or {}
        self.WORD = WORD
        self.word_completer = WordCompleter(words=self.words, WORD=self.WORD, meta_dict=self.meta_dict)
        self.fuzzy_completer = FuzzyCompleter(self.word_completer, WORD=self.WORD)

    def get_completions(self, document: Document, complete_event: CompleteEvent) -> Iterable[Completion]:
        return self.fuzzy_completer.get_completions(document, complete_event)