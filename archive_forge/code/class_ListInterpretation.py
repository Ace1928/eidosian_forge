import collections.abc
import contextlib
import sys
import textwrap
import weakref
from abc import ABC
from types import TracebackType
from weakref import ReferenceType
from debian._deb822_repro._util import (combine_into_replacement, BufferingIterator,
from debian._deb822_repro.formatter import (
from debian._deb822_repro.tokens import (
from debian._deb822_repro.types import AmbiguousDeb822FieldKeyError, SyntaxOrParseError
from debian._util import (
class ListInterpretation(GenericContentBasedInterpretation[Deb822ParsedTokenList[VE, ST], VE]):

    def __init__(self, tokenizer, value_parser, vtype, stype, default_separator_factory, render_factory):
        super().__init__(tokenizer, value_parser)
        self._vtype = vtype
        self._stype = stype
        self._default_separator_factory = default_separator_factory
        self._render_factory = render_factory

    def _high_level_interpretation(self, kvpair_element, token_list, discard_comments_on_read=True):
        return Deb822ParsedTokenList(kvpair_element, token_list, self._vtype, self._stype, self._parse_str, self._default_separator_factory, self._render_factory(discard_comments_on_read))