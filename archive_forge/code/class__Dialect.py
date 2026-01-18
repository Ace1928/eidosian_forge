from __future__ import annotations
import logging
import typing as t
from enum import Enum, auto
from functools import reduce
from sqlglot import exp
from sqlglot.errors import ParseError
from sqlglot.generator import Generator
from sqlglot.helper import AutoName, flatten, is_int, seq_get
from sqlglot.jsonpath import parse as parse_json_path
from sqlglot.parser import Parser
from sqlglot.time import TIMEZONES, format_time
from sqlglot.tokens import Token, Tokenizer, TokenType
from sqlglot.trie import new_trie
class _Dialect(type):
    classes: t.Dict[str, t.Type[Dialect]] = {}

    def __eq__(cls, other: t.Any) -> bool:
        if cls is other:
            return True
        if isinstance(other, str):
            return cls is cls.get(other)
        if isinstance(other, Dialect):
            return cls is type(other)
        return False

    def __hash__(cls) -> int:
        return hash(cls.__name__.lower())

    @classmethod
    def __getitem__(cls, key: str) -> t.Type[Dialect]:
        return cls.classes[key]

    @classmethod
    def get(cls, key: str, default: t.Optional[t.Type[Dialect]]=None) -> t.Optional[t.Type[Dialect]]:
        return cls.classes.get(key, default)

    def __new__(cls, clsname, bases, attrs):
        klass = super().__new__(cls, clsname, bases, attrs)
        enum = Dialects.__members__.get(clsname.upper())
        cls.classes[enum.value if enum is not None else clsname.lower()] = klass
        klass.TIME_TRIE = new_trie(klass.TIME_MAPPING)
        klass.FORMAT_TRIE = new_trie(klass.FORMAT_MAPPING) if klass.FORMAT_MAPPING else klass.TIME_TRIE
        klass.INVERSE_TIME_MAPPING = {v: k for k, v in klass.TIME_MAPPING.items()}
        klass.INVERSE_TIME_TRIE = new_trie(klass.INVERSE_TIME_MAPPING)
        base = seq_get(bases, 0)
        base_tokenizer = (getattr(base, 'tokenizer_class', Tokenizer),)
        base_parser = (getattr(base, 'parser_class', Parser),)
        base_generator = (getattr(base, 'generator_class', Generator),)
        klass.tokenizer_class = klass.__dict__.get('Tokenizer', type('Tokenizer', base_tokenizer, {}))
        klass.parser_class = klass.__dict__.get('Parser', type('Parser', base_parser, {}))
        klass.generator_class = klass.__dict__.get('Generator', type('Generator', base_generator, {}))
        klass.QUOTE_START, klass.QUOTE_END = list(klass.tokenizer_class._QUOTES.items())[0]
        klass.IDENTIFIER_START, klass.IDENTIFIER_END = list(klass.tokenizer_class._IDENTIFIERS.items())[0]

        def get_start_end(token_type: TokenType) -> t.Tuple[t.Optional[str], t.Optional[str]]:
            return next(((s, e) for s, (e, t) in klass.tokenizer_class._FORMAT_STRINGS.items() if t == token_type), (None, None))
        klass.BIT_START, klass.BIT_END = get_start_end(TokenType.BIT_STRING)
        klass.HEX_START, klass.HEX_END = get_start_end(TokenType.HEX_STRING)
        klass.BYTE_START, klass.BYTE_END = get_start_end(TokenType.BYTE_STRING)
        klass.UNICODE_START, klass.UNICODE_END = get_start_end(TokenType.UNICODE_STRING)
        if '\\' in klass.tokenizer_class.STRING_ESCAPES:
            klass.UNESCAPED_SEQUENCES = {'\\a': '\x07', '\\b': '\x08', '\\f': '\x0c', '\\n': '\n', '\\r': '\r', '\\t': '\t', '\\v': '\x0b', '\\\\': '\\', **klass.UNESCAPED_SEQUENCES}
        klass.ESCAPED_SEQUENCES = {v: k for k, v in klass.UNESCAPED_SEQUENCES.items()}
        if enum not in ('', 'bigquery'):
            klass.generator_class.SELECT_KINDS = ()
        if enum not in ('', 'databricks', 'hive', 'spark', 'spark2'):
            modifier_transforms = klass.generator_class.AFTER_HAVING_MODIFIER_TRANSFORMS.copy()
            for modifier in ('cluster', 'distribute', 'sort'):
                modifier_transforms.pop(modifier, None)
            klass.generator_class.AFTER_HAVING_MODIFIER_TRANSFORMS = modifier_transforms
        if not klass.SUPPORTS_SEMI_ANTI_JOIN:
            klass.parser_class.TABLE_ALIAS_TOKENS = klass.parser_class.TABLE_ALIAS_TOKENS | {TokenType.ANTI, TokenType.SEMI}
        return klass