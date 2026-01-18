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
class GenericContentBasedInterpretation(Interpretation[T], Generic[T, VE]):

    def __init__(self, tokenizer, value_parser):
        super().__init__()
        self._tokenizer = tokenizer
        self._value_parser = value_parser

    def _high_level_interpretation(self, kvpair_element, token_list, discard_comments_on_read=True):
        raise NotImplementedError

    def _parse_stream(self, buffered_iterator):
        value_parser = self._value_parser
        for token in buffered_iterator:
            if isinstance(token, Deb822ValueToken):
                yield value_parser(token, buffered_iterator)
            else:
                yield token

    def _parse_kvpair(self, kvpair):
        content = kvpair.value_element.convert_to_text()
        yield from self._parse_str(content)

    def _parse_str(self, content):
        content_len = len(content)
        biter = BufferingIterator(len_check_iterator(content, self._tokenizer(content), content_len=content_len))
        yield from len_check_iterator(content, self._parse_stream(biter), content_len=content_len)

    def interpret(self, kvpair_element, discard_comments_on_read=True):
        token_list = []
        token_list.extend(self._parse_kvpair(kvpair_element))
        return self._high_level_interpretation(kvpair_element, token_list, discard_comments_on_read=discard_comments_on_read)