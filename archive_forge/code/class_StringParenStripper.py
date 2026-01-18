import re
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import (
from mypy_extensions import trait
from black.comments import contains_pragma_comment
from black.lines import Line, append_leaves
from black.mode import Feature, Mode, Preview
from black.nodes import (
from black.rusty import Err, Ok, Result
from black.strings import (
from blib2to3.pgen2 import token
from blib2to3.pytree import Leaf, Node
class StringParenStripper(StringTransformer):
    """StringTransformer that strips surrounding parentheses from strings.

    Requirements:
        The line contains a string which is surrounded by parentheses and:
            - The target string is NOT the only argument to a function call.
            - The target string is NOT a "pointless" string.
            - If the target string contains a PERCENT, the brackets are not
              preceded or followed by an operator with higher precedence than
              PERCENT.

    Transformations:
        The parentheses mentioned in the 'Requirements' section are stripped.

    Collaborations:
        StringParenStripper has its own inherent usefulness, but it is also
        relied on to clean up the parentheses created by StringParenWrapper (in
        the event that they are no longer needed).
    """

    def do_match(self, line: Line) -> TMatchResult:
        LL = line.leaves
        is_valid_index = is_valid_index_factory(LL)
        string_indices = []
        idx = -1
        while True:
            idx += 1
            if idx >= len(LL):
                break
            leaf = LL[idx]
            if leaf.type != token.STRING:
                continue
            if leaf.parent and leaf.parent.parent and (leaf.parent.parent.type == syms.simple_stmt):
                continue
            if not is_valid_index(idx - 1) or LL[idx - 1].type != token.LPAR or is_empty_lpar(LL[idx - 1]):
                continue
            if is_valid_index(idx - 2) and (LL[idx - 2].type == token.NAME or LL[idx - 2].type in CLOSING_BRACKETS):
                continue
            string_idx = idx
            string_parser = StringParser()
            next_idx = string_parser.parse(LL, string_idx)
            if is_valid_index(idx - 2):
                before_lpar = LL[idx - 2]
                if token.PERCENT in {leaf.type for leaf in LL[idx - 1:next_idx]} and (before_lpar.type in {token.STAR, token.AT, token.SLASH, token.DOUBLESLASH, token.PERCENT, token.TILDE, token.DOUBLESTAR, token.AWAIT, token.LSQB, token.LPAR} or (before_lpar.parent and before_lpar.parent.type == syms.factor and (before_lpar.type in {token.PLUS, token.MINUS}))):
                    continue
            if is_valid_index(next_idx) and LL[next_idx].type == token.RPAR and (not is_empty_rpar(LL[next_idx])):
                if is_valid_index(next_idx + 1) and LL[next_idx + 1].type in {token.DOUBLESTAR, token.LSQB, token.LPAR, token.DOT}:
                    continue
                string_indices.append(string_idx)
                idx = string_idx
                while idx < len(LL) - 1 and LL[idx + 1].type == token.STRING:
                    idx += 1
        if string_indices:
            return Ok(string_indices)
        return TErr('This line has no strings wrapped in parens.')

    def do_transform(self, line: Line, string_indices: List[int]) -> Iterator[TResult[Line]]:
        LL = line.leaves
        string_and_rpar_indices: List[int] = []
        for string_idx in string_indices:
            string_parser = StringParser()
            rpar_idx = string_parser.parse(LL, string_idx)
            should_transform = True
            for leaf in (LL[string_idx - 1], LL[rpar_idx]):
                if line.comments_after(leaf):
                    should_transform = False
                    break
            if should_transform:
                string_and_rpar_indices.extend((string_idx, rpar_idx))
        if string_and_rpar_indices:
            yield Ok(self._transform_to_new_line(line, string_and_rpar_indices))
        else:
            yield Err(CannotTransform('All string groups have comments attached to them.'))

    def _transform_to_new_line(self, line: Line, string_and_rpar_indices: List[int]) -> Line:
        LL = line.leaves
        new_line = line.clone()
        new_line.comments = line.comments.copy()
        previous_idx = -1
        for idx in sorted(string_and_rpar_indices):
            leaf = LL[idx]
            lpar_or_rpar_idx = idx - 1 if leaf.type == token.STRING else idx
            append_leaves(new_line, line, LL[previous_idx + 1:lpar_or_rpar_idx])
            if leaf.type == token.STRING:
                string_leaf = Leaf(token.STRING, LL[idx].value)
                LL[lpar_or_rpar_idx].remove()
                replace_child(LL[idx], string_leaf)
                new_line.append(string_leaf)
                old_comments = new_line.comments.pop(id(LL[idx]), [])
                new_line.comments.setdefault(id(string_leaf), []).extend(old_comments)
            else:
                LL[lpar_or_rpar_idx].remove()
            previous_idx = idx
        append_leaves(new_line, line, LL[idx + 1:])
        return new_line