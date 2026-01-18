from __future__ import annotations
from abc import ABCMeta, abstractmethod
from typing import Callable, Hashable
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text.base import StyleAndTextTuples
class DynamicLexer(Lexer):
    """
    Lexer class that can dynamically returns any Lexer.

    :param get_lexer: Callable that returns a :class:`.Lexer` instance.
    """

    def __init__(self, get_lexer: Callable[[], Lexer | None]) -> None:
        self.get_lexer = get_lexer
        self._dummy = SimpleLexer()

    def lex_document(self, document: Document) -> Callable[[int], StyleAndTextTuples]:
        lexer = self.get_lexer() or self._dummy
        return lexer.lex_document(document)

    def invalidation_hash(self) -> Hashable:
        lexer = self.get_lexer() or self._dummy
        return id(lexer)