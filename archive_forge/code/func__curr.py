from __future__ import annotations
import typing as t
import sqlglot.expressions as exp
from sqlglot.errors import ParseError
from sqlglot.tokens import Token, Tokenizer, TokenType
def _curr() -> t.Optional[TokenType]:
    return tokens[i].token_type if i < size else None