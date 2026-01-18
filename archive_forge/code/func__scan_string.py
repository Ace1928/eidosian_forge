from __future__ import annotations
import os
import typing as t
from enum import auto
from sqlglot.errors import SqlglotError, TokenError
from sqlglot.helper import AutoName
from sqlglot.trie import TrieResult, in_trie, new_trie
def _scan_string(self, start: str) -> bool:
    base = None
    token_type = TokenType.STRING
    if start in self._QUOTES:
        end = self._QUOTES[start]
    elif start in self._FORMAT_STRINGS:
        end, token_type = self._FORMAT_STRINGS[start]
        if token_type == TokenType.HEX_STRING:
            base = 16
        elif token_type == TokenType.BIT_STRING:
            base = 2
        elif token_type == TokenType.HEREDOC_STRING:
            if self.HEREDOC_TAG_IS_IDENTIFIER and (not self._peek.isidentifier()) and (not self._peek == end):
                if self.HEREDOC_STRING_ALTERNATIVE != token_type.VAR:
                    self._add(self.HEREDOC_STRING_ALTERNATIVE)
                else:
                    self._scan_var()
                return True
            self._advance()
            if self._char == end:
                tag = ''
            else:
                tag = self._extract_string(end, unescape_sequences=False, raise_unmatched=not self.HEREDOC_TAG_IS_IDENTIFIER)
            if self._end and tag and self.HEREDOC_TAG_IS_IDENTIFIER:
                self._advance(-len(tag))
                self._add(self.HEREDOC_STRING_ALTERNATIVE)
                return True
            end = f'{start}{tag}{end}'
    else:
        return False
    self._advance(len(start))
    text = self._extract_string(end, unescape_sequences=token_type != TokenType.RAW_STRING)
    if base:
        try:
            int(text, base)
        except Exception:
            raise TokenError(f'Numeric string contains invalid characters from {self._line}:{self._start}')
    self._add(token_type, text)
    return True