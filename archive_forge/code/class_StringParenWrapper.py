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
class StringParenWrapper(BaseStringSplitter, CustomSplitMapMixin):
    """
    StringTransformer that wraps strings in parens and then splits at the LPAR.

    Requirements:
        All of the requirements listed in BaseStringSplitter's docstring in
        addition to the requirements listed below:

        * The line is a return/yield statement, which returns/yields a string.
          OR
        * The line is part of a ternary expression (e.g. `x = y if cond else
          z`) such that the line starts with `else <string>`, where <string> is
          some string.
          OR
        * The line is an assert statement, which ends with a string.
          OR
        * The line is an assignment statement (e.g. `x = <string>` or `x +=
          <string>`) such that the variable is being assigned the value of some
          string.
          OR
        * The line is a dictionary key assignment where some valid key is being
          assigned the value of some string.
          OR
        * The line is an lambda expression and the value is a string.
          OR
        * The line starts with an "atom" string that prefers to be wrapped in
          parens. It's preferred to be wrapped when it's is an immediate child of
          a list/set/tuple literal, AND the string is surrounded by commas (or is
          the first/last child).

    Transformations:
        The chosen string is wrapped in parentheses and then split at the LPAR.

        We then have one line which ends with an LPAR and another line that
        starts with the chosen string. The latter line is then split again at
        the RPAR. This results in the RPAR (and possibly a trailing comma)
        being placed on its own line.

        NOTE: If any leaves exist to the right of the chosen string (except
        for a trailing comma, which would be placed after the RPAR), those
        leaves are placed inside the parentheses.  In effect, the chosen
        string is not necessarily being "wrapped" by parentheses. We can,
        however, count on the LPAR being placed directly before the chosen
        string.

        In other words, StringParenWrapper creates "atom" strings. These
        can then be split again by StringSplitter, if necessary.

    Collaborations:
        In the event that a string line split by StringParenWrapper is
        changed such that it no longer needs to be given its own line,
        StringParenWrapper relies on StringParenStripper to clean up the
        parentheses it created.

        For "atom" strings that prefers to be wrapped in parens, it requires
        StringSplitter to hold the split until the string is wrapped in parens.
    """

    def do_splitter_match(self, line: Line) -> TMatchResult:
        LL = line.leaves
        if line.leaves[-1].type in OPENING_BRACKETS:
            return TErr('Cannot wrap parens around a line that ends in an opening bracket.')
        string_idx = self._return_match(LL) or self._else_match(LL) or self._assert_match(LL) or self._assign_match(LL) or self._dict_or_lambda_match(LL) or self._prefer_paren_wrap_match(LL)
        if string_idx is not None:
            string_value = line.leaves[string_idx].value
            if not any((char == ' ' or char in SPLIT_SAFE_CHARS for char in string_value)):
                max_string_width = self.line_length - (line.depth + 1) * 4
                if str_width(string_value) > max_string_width:
                    if not self.has_custom_splits(string_value):
                        return TErr("We do not wrap long strings in parentheses when the resultant line would still be over the specified line length and can't be split further by StringSplitter.")
            return Ok([string_idx])
        return TErr('This line does not contain any non-atomic strings.')

    @staticmethod
    def _return_match(LL: List[Leaf]) -> Optional[int]:
        """
        Returns:
            string_idx such that @LL[string_idx] is equal to our target (i.e.
            matched) string, if this line matches the return/yield statement
            requirements listed in the 'Requirements' section of this classes'
            docstring.
                OR
            None, otherwise.
        """
        if parent_type(LL[0]) in [syms.return_stmt, syms.yield_expr] and LL[0].value in ['return', 'yield']:
            is_valid_index = is_valid_index_factory(LL)
            idx = 2 if is_valid_index(1) and is_empty_par(LL[1]) else 1
            if is_valid_index(idx) and LL[idx].type == token.STRING:
                return idx
        return None

    @staticmethod
    def _else_match(LL: List[Leaf]) -> Optional[int]:
        """
        Returns:
            string_idx such that @LL[string_idx] is equal to our target (i.e.
            matched) string, if this line matches the ternary expression
            requirements listed in the 'Requirements' section of this classes'
            docstring.
                OR
            None, otherwise.
        """
        if parent_type(LL[0]) == syms.test and LL[0].type == token.NAME and (LL[0].value == 'else'):
            is_valid_index = is_valid_index_factory(LL)
            idx = 2 if is_valid_index(1) and is_empty_par(LL[1]) else 1
            if is_valid_index(idx) and LL[idx].type == token.STRING:
                return idx
        return None

    @staticmethod
    def _assert_match(LL: List[Leaf]) -> Optional[int]:
        """
        Returns:
            string_idx such that @LL[string_idx] is equal to our target (i.e.
            matched) string, if this line matches the assert statement
            requirements listed in the 'Requirements' section of this classes'
            docstring.
                OR
            None, otherwise.
        """
        if parent_type(LL[0]) == syms.assert_stmt and LL[0].value == 'assert':
            is_valid_index = is_valid_index_factory(LL)
            for i, leaf in enumerate(LL):
                if leaf.type == token.COMMA:
                    idx = i + 2 if is_empty_par(LL[i + 1]) else i + 1
                    if is_valid_index(idx) and LL[idx].type == token.STRING:
                        string_idx = idx
                        string_parser = StringParser()
                        idx = string_parser.parse(LL, string_idx)
                        if not is_valid_index(idx):
                            return string_idx
        return None

    @staticmethod
    def _assign_match(LL: List[Leaf]) -> Optional[int]:
        """
        Returns:
            string_idx such that @LL[string_idx] is equal to our target (i.e.
            matched) string, if this line matches the assignment statement
            requirements listed in the 'Requirements' section of this classes'
            docstring.
                OR
            None, otherwise.
        """
        if parent_type(LL[0]) in [syms.expr_stmt, syms.argument, syms.power] and LL[0].type == token.NAME:
            is_valid_index = is_valid_index_factory(LL)
            for i, leaf in enumerate(LL):
                if leaf.type in [token.EQUAL, token.PLUSEQUAL]:
                    idx = i + 2 if is_empty_par(LL[i + 1]) else i + 1
                    if is_valid_index(idx) and LL[idx].type == token.STRING:
                        string_idx = idx
                        string_parser = StringParser()
                        idx = string_parser.parse(LL, string_idx)
                        if parent_type(LL[0]) == syms.argument and is_valid_index(idx) and (LL[idx].type == token.COMMA):
                            idx += 1
                        if not is_valid_index(idx):
                            return string_idx
        return None

    @staticmethod
    def _dict_or_lambda_match(LL: List[Leaf]) -> Optional[int]:
        """
        Returns:
            string_idx such that @LL[string_idx] is equal to our target (i.e.
            matched) string, if this line matches the dictionary key assignment
            statement or lambda expression requirements listed in the
            'Requirements' section of this classes' docstring.
                OR
            None, otherwise.
        """
        parent_types = [parent_type(LL[0]), parent_type(LL[0].parent)]
        if syms.dictsetmaker in parent_types or syms.lambdef in parent_types:
            is_valid_index = is_valid_index_factory(LL)
            for i, leaf in enumerate(LL):
                if leaf.type == token.COLON and i < len(LL) - 1:
                    idx = i + 2 if is_empty_par(LL[i + 1]) else i + 1
                    if is_valid_index(idx) and LL[idx].type == token.STRING:
                        string_idx = idx
                        string_parser = StringParser()
                        idx = string_parser.parse(LL, string_idx)
                        if is_valid_index(idx) and LL[idx].type == token.COMMA:
                            idx += 1
                        if not is_valid_index(idx):
                            return string_idx
        return None

    def do_transform(self, line: Line, string_indices: List[int]) -> Iterator[TResult[Line]]:
        LL = line.leaves
        assert len(string_indices) == 1, f'{self.__class__.__name__} should only find one match at a time, found {len(string_indices)}'
        string_idx = string_indices[0]
        is_valid_index = is_valid_index_factory(LL)
        insert_str_child = insert_str_child_factory(LL[string_idx])
        comma_idx = -1
        ends_with_comma = False
        if LL[comma_idx].type == token.COMMA:
            ends_with_comma = True
        leaves_to_steal_comments_from = [LL[string_idx]]
        if ends_with_comma:
            leaves_to_steal_comments_from.append(LL[comma_idx])
        first_line = line.clone()
        left_leaves = LL[:string_idx]
        old_parens_exist = False
        if left_leaves and left_leaves[-1].type == token.LPAR:
            old_parens_exist = True
            leaves_to_steal_comments_from.append(left_leaves[-1])
            left_leaves.pop()
        append_leaves(first_line, line, left_leaves)
        lpar_leaf = Leaf(token.LPAR, '(')
        if old_parens_exist:
            replace_child(LL[string_idx - 1], lpar_leaf)
        else:
            insert_str_child(lpar_leaf)
        first_line.append(lpar_leaf)
        for leaf in leaves_to_steal_comments_from:
            for comment_leaf in line.comments_after(leaf):
                first_line.append(comment_leaf, preformatted=True)
        yield Ok(first_line)
        string_value = LL[string_idx].value
        string_line = Line(mode=line.mode, depth=line.depth + 1, inside_brackets=True, should_split_rhs=line.should_split_rhs, magic_trailing_comma=line.magic_trailing_comma)
        string_leaf = Leaf(token.STRING, string_value)
        insert_str_child(string_leaf)
        string_line.append(string_leaf)
        old_rpar_leaf = None
        if is_valid_index(string_idx + 1):
            right_leaves = LL[string_idx + 1:]
            if ends_with_comma:
                right_leaves.pop()
            if old_parens_exist:
                assert right_leaves and right_leaves[-1].type == token.RPAR, f'Apparently, old parentheses do NOT exist?! (left_leaves={left_leaves}, right_leaves={right_leaves})'
                old_rpar_leaf = right_leaves.pop()
            elif right_leaves and right_leaves[-1].type == token.RPAR:
                opening_bracket = right_leaves[-1].opening_bracket
                if opening_bracket is not None and opening_bracket in left_leaves:
                    index = left_leaves.index(opening_bracket)
                    if 0 < index < len(left_leaves) - 1 and left_leaves[index - 1].type == token.COLON and (left_leaves[index + 1].value == 'lambda'):
                        right_leaves.pop()
            append_leaves(string_line, line, right_leaves)
        yield Ok(string_line)
        last_line = line.clone()
        last_line.bracket_tracker = first_line.bracket_tracker
        new_rpar_leaf = Leaf(token.RPAR, ')')
        if old_rpar_leaf is not None:
            replace_child(old_rpar_leaf, new_rpar_leaf)
        else:
            insert_str_child(new_rpar_leaf)
        last_line.append(new_rpar_leaf)
        if ends_with_comma:
            comma_leaf = Leaf(token.COMMA, ',')
            replace_child(LL[comma_idx], comma_leaf)
            last_line.append(comma_leaf)
        yield Ok(last_line)