from __future__ import annotations
import logging
import typing as t
from collections import defaultdict
from sqlglot import exp
from sqlglot.errors import ErrorLevel, ParseError, concat_messages, merge_errors
from sqlglot.helper import apply_index_offset, ensure_list, seq_get
from sqlglot.time import format_time
from sqlglot.tokens import Token, Tokenizer, TokenType
from sqlglot.trie import TrieResult, in_trie, new_trie
def _parse_function_call(self, functions: t.Optional[t.Dict[str, t.Callable]]=None, anonymous: bool=False, optional_parens: bool=True, any_token: bool=False) -> t.Optional[exp.Expression]:
    if not self._curr:
        return None
    comments = self._curr.comments
    token_type = self._curr.token_type
    this = self._curr.text
    upper = this.upper()
    parser = self.NO_PAREN_FUNCTION_PARSERS.get(upper)
    if optional_parens and parser and (token_type not in self.INVALID_FUNC_NAME_TOKENS):
        self._advance()
        return self._parse_window(parser(self))
    if not self._next or self._next.token_type != TokenType.L_PAREN:
        if optional_parens and token_type in self.NO_PAREN_FUNCTIONS:
            self._advance()
            return self.expression(self.NO_PAREN_FUNCTIONS[token_type])
        return None
    if any_token:
        if token_type in self.RESERVED_TOKENS and token_type != TokenType.IDENTIFIER:
            return None
    elif token_type not in self.FUNC_TOKENS:
        return None
    self._advance(2)
    parser = self.FUNCTION_PARSERS.get(upper)
    if parser and (not anonymous):
        this = parser(self)
    else:
        subquery_predicate = self.SUBQUERY_PREDICATES.get(token_type)
        if subquery_predicate and self._curr.token_type in (TokenType.SELECT, TokenType.WITH):
            this = self.expression(subquery_predicate, this=self._parse_select())
            self._match_r_paren()
            return this
        if functions is None:
            functions = self.FUNCTIONS
        function = functions.get(upper)
        alias = upper in self.FUNCTIONS_WITH_ALIASED_ARGS
        args = self._parse_csv(lambda: self._parse_lambda(alias=alias))
        if alias:
            args = self._kv_to_prop_eq(args)
        if function and (not anonymous):
            if 'dialect' in function.__code__.co_varnames:
                func = function(args, dialect=self.dialect)
            else:
                func = function(args)
            func = self.validate_expression(func, args)
            if not self.dialect.NORMALIZE_FUNCTIONS:
                func.meta['name'] = this
            this = func
        else:
            if token_type == TokenType.IDENTIFIER:
                this = exp.Identifier(this=this, quoted=True)
            this = self.expression(exp.Anonymous, this=this, expressions=args)
    if isinstance(this, exp.Expression):
        this.add_comments(comments)
    self._match_r_paren(this)
    return self._parse_window(this)