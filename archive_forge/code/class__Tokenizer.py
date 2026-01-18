from __future__ import annotations
import os
import typing as t
from enum import auto
from sqlglot.errors import SqlglotError, TokenError
from sqlglot.helper import AutoName
from sqlglot.trie import TrieResult, in_trie, new_trie
class _Tokenizer(type):

    def __new__(cls, clsname, bases, attrs):
        klass = super().__new__(cls, clsname, bases, attrs)

        def _convert_quotes(arr: t.List[str | t.Tuple[str, str]]) -> t.Dict[str, str]:
            return dict(((item, item) if isinstance(item, str) else (item[0], item[1]) for item in arr))

        def _quotes_to_format(token_type: TokenType, arr: t.List[str | t.Tuple[str, str]]) -> t.Dict[str, t.Tuple[str, TokenType]]:
            return {k: (v, token_type) for k, v in _convert_quotes(arr).items()}
        klass._QUOTES = _convert_quotes(klass.QUOTES)
        klass._IDENTIFIERS = _convert_quotes(klass.IDENTIFIERS)
        klass._FORMAT_STRINGS = {**{p + s: (e, TokenType.NATIONAL_STRING) for s, e in klass._QUOTES.items() for p in ('n', 'N')}, **_quotes_to_format(TokenType.BIT_STRING, klass.BIT_STRINGS), **_quotes_to_format(TokenType.BYTE_STRING, klass.BYTE_STRINGS), **_quotes_to_format(TokenType.HEX_STRING, klass.HEX_STRINGS), **_quotes_to_format(TokenType.RAW_STRING, klass.RAW_STRINGS), **_quotes_to_format(TokenType.HEREDOC_STRING, klass.HEREDOC_STRINGS), **_quotes_to_format(TokenType.UNICODE_STRING, klass.UNICODE_STRINGS)}
        klass._STRING_ESCAPES = set(klass.STRING_ESCAPES)
        klass._IDENTIFIER_ESCAPES = set(klass.IDENTIFIER_ESCAPES)
        klass._COMMENTS = {**dict(((comment, None) if isinstance(comment, str) else (comment[0], comment[1]) for comment in klass.COMMENTS)), '{#': '#}'}
        klass._KEYWORD_TRIE = new_trie((key.upper() for key in (*klass.KEYWORDS, *klass._COMMENTS, *klass._QUOTES, *klass._FORMAT_STRINGS) if ' ' in key or any((single in key for single in klass.SINGLE_TOKENS))))
        if USE_RS_TOKENIZER:
            settings = RsTokenizerSettings(white_space={k: _TOKEN_TYPE_TO_INDEX[v] for k, v in klass.WHITE_SPACE.items()}, single_tokens={k: _TOKEN_TYPE_TO_INDEX[v] for k, v in klass.SINGLE_TOKENS.items()}, keywords={k: _TOKEN_TYPE_TO_INDEX[v] for k, v in klass.KEYWORDS.items()}, numeric_literals=klass.NUMERIC_LITERALS, identifiers=klass._IDENTIFIERS, identifier_escapes=klass._IDENTIFIER_ESCAPES, string_escapes=klass._STRING_ESCAPES, quotes=klass._QUOTES, format_strings={k: (v1, _TOKEN_TYPE_TO_INDEX[v2]) for k, (v1, v2) in klass._FORMAT_STRINGS.items()}, has_bit_strings=bool(klass.BIT_STRINGS), has_hex_strings=bool(klass.HEX_STRINGS), comments=klass._COMMENTS, var_single_tokens=klass.VAR_SINGLE_TOKENS, commands={_TOKEN_TYPE_TO_INDEX[v] for v in klass.COMMANDS}, command_prefix_tokens={_TOKEN_TYPE_TO_INDEX[v] for v in klass.COMMAND_PREFIX_TOKENS}, heredoc_tag_is_identifier=klass.HEREDOC_TAG_IS_IDENTIFIER)
            token_types = RsTokenTypeSettings(bit_string=_TOKEN_TYPE_TO_INDEX[TokenType.BIT_STRING], break_=_TOKEN_TYPE_TO_INDEX[TokenType.BREAK], dcolon=_TOKEN_TYPE_TO_INDEX[TokenType.DCOLON], heredoc_string=_TOKEN_TYPE_TO_INDEX[TokenType.HEREDOC_STRING], raw_string=_TOKEN_TYPE_TO_INDEX[TokenType.RAW_STRING], hex_string=_TOKEN_TYPE_TO_INDEX[TokenType.HEX_STRING], identifier=_TOKEN_TYPE_TO_INDEX[TokenType.IDENTIFIER], number=_TOKEN_TYPE_TO_INDEX[TokenType.NUMBER], parameter=_TOKEN_TYPE_TO_INDEX[TokenType.PARAMETER], semicolon=_TOKEN_TYPE_TO_INDEX[TokenType.SEMICOLON], string=_TOKEN_TYPE_TO_INDEX[TokenType.STRING], var=_TOKEN_TYPE_TO_INDEX[TokenType.VAR], heredoc_string_alternative=_TOKEN_TYPE_TO_INDEX[klass.HEREDOC_STRING_ALTERNATIVE])
            klass._RS_TOKENIZER = RsTokenizer(settings, token_types)
        else:
            klass._RS_TOKENIZER = None
        return klass