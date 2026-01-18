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
def _parser_to_value_factory(parser, vtype):

    def _value_factory(v):
        if v == '':
            raise ValueError('The empty string is not a value')
        token_iter = iter(parser(v))
        t1 = next(token_iter, None)
        t2 = next(token_iter, None)
        assert t1 is not None, 'Bad parser - it returned None (or no TE) for "' + v + '"'
        if t2 is not None:
            msg = textwrap.dedent('            The input "{v}" should have been exactly one element, but the parser provided at\n             least two.  This can happen with unnecessary leading/trailing whitespace\n             or including commas the value for a comma list.\n            ').format(v=v)
            raise ValueError(msg)
        if not isinstance(t1, vtype):
            if isinstance(t1, Deb822Token) and (t1.is_comment or t1.is_whitespace):
                raise ValueError('The input "{v}" is whitespace or a comment: Expected a value')
            msg = 'The input "{v}" should have produced a element of type {vtype_name}, but instead it produced {t1}'
            raise ValueError(msg.format(v=v, vtype_name=vtype.__name__, t1=t1))
        assert len(t1.convert_to_text()) == len(v), 'Bad tokenizer - the token did not cover the input text exactly ({t1_len} != {v_len}'.format(t1_len=len(t1.convert_to_text()), v_len=len(v))
        return t1
    return _value_factory