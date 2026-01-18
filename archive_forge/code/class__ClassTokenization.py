import collections.abc
import dataclasses
import functools
import inspect
import io
import itertools
import tokenize
from typing import Callable, Dict, Generic, Hashable, List, Optional, Type, TypeVar
import docstring_parser
from typing_extensions import get_origin, is_typeddict
from . import _resolver, _strings, _unsafe_cache
@dataclasses.dataclass(frozen=True)
class _ClassTokenization:
    tokens: List[_Token]
    tokens_from_logical_line: Dict[int, List[_Token]]
    tokens_from_actual_line: Dict[int, List[_Token]]
    field_data_from_name: Dict[str, _FieldData]

    @staticmethod
    @_unsafe_cache.unsafe_cache(64)
    def make(clz) -> '_ClassTokenization':
        """Parse the source code of a class, and cache some tokenization information."""
        readline = io.BytesIO(inspect.getsource(clz).encode('utf-8')).readline
        tokens: List[_Token] = []
        tokens_from_logical_line: Dict[int, List[_Token]] = {1: []}
        tokens_from_actual_line: Dict[int, List[_Token]] = {1: []}
        field_data_from_name: Dict[str, _FieldData] = {}
        logical_line: int = 1
        actual_line: int = 1
        for toktype, tok, start, end, line in tokenize.tokenize(readline):
            if toktype == tokenize.NEWLINE:
                logical_line += 1
                actual_line += 1
                tokens_from_logical_line[logical_line] = []
                tokens_from_actual_line[actual_line] = []
            elif toktype == tokenize.NL:
                actual_line += 1
                tokens_from_actual_line[actual_line] = []
            elif toktype is not tokenize.INDENT:
                token = _Token(token_type=toktype, content=tok, logical_line=logical_line, actual_line=actual_line)
                tokens.append(token)
                tokens_from_logical_line[logical_line].append(token)
                tokens_from_actual_line[actual_line].append(token)
        prev_field_logical_line: int = 1
        for i, token in enumerate(tokens[:-1]):
            if token.token_type == tokenize.NAME:
                is_first_token = True
                for t in tokens_from_logical_line[token.logical_line]:
                    if t == token:
                        break
                    if t.token_type is not tokenize.COMMENT:
                        is_first_token = False
                        break
                if not is_first_token:
                    continue
                if tokens[i + 1].content == ':' and token.content not in field_data_from_name:
                    field_data_from_name[token.content] = _FieldData(index=i, logical_line=token.logical_line, actual_line=token.actual_line, prev_field_logical_line=prev_field_logical_line)
                    prev_field_logical_line = token.logical_line
        return _ClassTokenization(tokens=tokens, tokens_from_logical_line=tokens_from_logical_line, tokens_from_actual_line=tokens_from_actual_line, field_data_from_name=field_data_from_name)