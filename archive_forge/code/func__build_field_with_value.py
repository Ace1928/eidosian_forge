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
def _build_field_with_value(token_stream):
    buffered_stream = BufferingIterator(token_stream)
    for token_or_element in buffered_stream:
        start_of_field = False
        comment_element = None
        if isinstance(token_or_element, Deb822FieldNameToken):
            start_of_field = True
        elif isinstance(token_or_element, Deb822CommentElement):
            comment_element = token_or_element
            next_token = buffered_stream.peek()
            start_of_field = isinstance(next_token, Deb822FieldNameToken)
            if start_of_field:
                try:
                    token_or_element = next(buffered_stream)
                except StopIteration:
                    raise AssertionError
        if start_of_field:
            field_name = token_or_element
            separator = next(buffered_stream, None)
            value_element = next(buffered_stream, None)
            if separator is None or value_element is None:
                if comment_element:
                    yield comment_element
                error_elements = [field_name]
                if separator is not None:
                    error_elements.append(separator)
                yield Deb822ErrorElement(error_elements)
                return
            if isinstance(separator, Deb822FieldSeparatorToken) and isinstance(value_element, Deb822ValueElement):
                yield Deb822KeyValuePairElement(comment_element, cast('Deb822FieldNameToken', field_name), separator, value_element)
            else:
                error_tokens = [token_or_element]
                error_tokens.extend(buffered_stream.takewhile(_non_end_of_line_token))
                nl = buffered_stream.peek()
                if nl and isinstance(nl, Deb822NewlineAfterValueToken):
                    next(buffered_stream, None)
                    error_tokens.append(nl)
                yield Deb822ErrorElement(error_tokens)
        else:
            yield token_or_element