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
class StringSplitter(BaseStringSplitter, CustomSplitMapMixin):
    """
    StringTransformer that splits "atom" strings (i.e. strings which exist on
    lines by themselves).

    Requirements:
        * The line consists ONLY of a single string (possibly prefixed by a
          string operator [e.g. '+' or '==']), MAYBE a string trailer, and MAYBE
          a trailing comma.
          AND
        * All of the requirements listed in BaseStringSplitter's docstring.

    Transformations:
        The string mentioned in the 'Requirements' section is split into as
        many substrings as necessary to adhere to the configured line length.

        In the final set of substrings, no substring should be smaller than
        MIN_SUBSTR_SIZE characters.

        The string will ONLY be split on spaces (i.e. each new substring should
        start with a space). Note that the string will NOT be split on a space
        which is escaped with a backslash.

        If the string is an f-string, it will NOT be split in the middle of an
        f-expression (e.g. in f"FooBar: {foo() if x else bar()}", {foo() if x
        else bar()} is an f-expression).

        If the string that is being split has an associated set of custom split
        records and those custom splits will NOT result in any line going over
        the configured line length, those custom splits are used. Otherwise the
        string is split as late as possible (from left-to-right) while still
        adhering to the transformation rules listed above.

    Collaborations:
        StringSplitter relies on StringMerger to construct the appropriate
        CustomSplit objects and add them to the custom split map.
    """
    MIN_SUBSTR_SIZE: Final = 6

    def do_splitter_match(self, line: Line) -> TMatchResult:
        LL = line.leaves
        if self._prefer_paren_wrap_match(LL) is not None:
            return TErr('Line needs to be wrapped in parens first.')
        is_valid_index = is_valid_index_factory(LL)
        idx = 0
        if is_valid_index(idx) and is_valid_index(idx + 1) and ([LL[idx].type, LL[idx + 1].type] == [token.NAME, token.NAME]) and (str(LL[idx]) + str(LL[idx + 1]) == 'not in'):
            idx += 2
        elif is_valid_index(idx) and (LL[idx].type in self.STRING_OPERATORS or (LL[idx].type == token.NAME and str(LL[idx]) == 'in')):
            idx += 1
        if is_valid_index(idx) and is_empty_lpar(LL[idx]):
            idx += 1
        if not is_valid_index(idx) or LL[idx].type != token.STRING:
            return TErr('Line does not start with a string.')
        string_idx = idx
        string_parser = StringParser()
        idx = string_parser.parse(LL, string_idx)
        if is_valid_index(idx) and is_empty_rpar(LL[idx]):
            idx += 1
        if is_valid_index(idx) and LL[idx].type == token.COMMA:
            idx += 1
        if is_valid_index(idx):
            return TErr('This line does not end with a string.')
        return Ok([string_idx])

    def do_transform(self, line: Line, string_indices: List[int]) -> Iterator[TResult[Line]]:
        LL = line.leaves
        assert len(string_indices) == 1, f'{self.__class__.__name__} should only find one match at a time, found {len(string_indices)}'
        string_idx = string_indices[0]
        QUOTE = LL[string_idx].value[-1]
        is_valid_index = is_valid_index_factory(LL)
        insert_str_child = insert_str_child_factory(LL[string_idx])
        prefix = get_string_prefix(LL[string_idx].value).lower()
        drop_pointless_f_prefix = 'f' in prefix and fstring_contains_expr(LL[string_idx].value)
        first_string_line = True
        string_op_leaves = self._get_string_operator_leaves(LL)
        string_op_leaves_length = sum((len(str(prefix_leaf)) for prefix_leaf in string_op_leaves)) + 1 if string_op_leaves else 0

        def maybe_append_string_operators(new_line: Line) -> None:
            """
            Side Effects:
                If @line starts with a string operator and this is the first
                line we are constructing, this function appends the string
                operator to @new_line and replaces the old string operator leaf
                in the node structure. Otherwise this function does nothing.
            """
            maybe_prefix_leaves = string_op_leaves if first_string_line else []
            for i, prefix_leaf in enumerate(maybe_prefix_leaves):
                replace_child(LL[i], prefix_leaf)
                new_line.append(prefix_leaf)
        ends_with_comma = is_valid_index(string_idx + 1) and LL[string_idx + 1].type == token.COMMA

        def max_last_string_column() -> int:
            """
            Returns:
                The max allowed width of the string value used for the last
                line we will construct.  Note that this value means the width
                rather than the number of characters (e.g., many East Asian
                characters expand to two columns).
            """
            result = self.line_length
            result -= line.depth * 4
            result -= 1 if ends_with_comma else 0
            result -= string_op_leaves_length
            return result
        max_break_width = self.line_length
        max_break_width -= 1
        max_break_width -= line.depth * 4
        if max_break_width < 0:
            yield TErr(f'Unable to split {LL[string_idx].value} at such high of a line depth: {line.depth}')
            return
        custom_splits = self.pop_custom_splits(LL[string_idx].value)
        use_custom_breakpoints = bool(custom_splits and all((csplit.break_idx <= max_break_width for csplit in custom_splits)))
        rest_value = LL[string_idx].value

        def more_splits_should_be_made() -> bool:
            """
            Returns:
                True iff `rest_value` (the remaining string value from the last
                split), should be split again.
            """
            if use_custom_breakpoints:
                return len(custom_splits) > 1
            else:
                return str_width(rest_value) > max_last_string_column()
        string_line_results: List[Ok[Line]] = []
        while more_splits_should_be_made():
            if use_custom_breakpoints:
                csplit = custom_splits.pop(0)
                break_idx = csplit.break_idx
            else:
                max_bidx = count_chars_in_width(rest_value, max_break_width) - string_op_leaves_length
                maybe_break_idx = self._get_break_idx(rest_value, max_bidx)
                if maybe_break_idx is None:
                    if custom_splits:
                        rest_value = LL[string_idx].value
                        string_line_results = []
                        first_string_line = True
                        use_custom_breakpoints = True
                        continue
                    break
                break_idx = maybe_break_idx
            next_value = rest_value[:break_idx] + QUOTE
            if use_custom_breakpoints and (not csplit.has_prefix) and (next_value == prefix + QUOTE or next_value != self._normalize_f_string(next_value, prefix)):
                break_idx += 1
                next_value = rest_value[:break_idx] + QUOTE
            if drop_pointless_f_prefix:
                next_value = self._normalize_f_string(next_value, prefix)
            next_leaf = Leaf(token.STRING, next_value)
            insert_str_child(next_leaf)
            self._maybe_normalize_string_quotes(next_leaf)
            next_line = line.clone()
            maybe_append_string_operators(next_line)
            next_line.append(next_leaf)
            string_line_results.append(Ok(next_line))
            rest_value = prefix + QUOTE + rest_value[break_idx:]
            first_string_line = False
        yield from string_line_results
        if drop_pointless_f_prefix:
            rest_value = self._normalize_f_string(rest_value, prefix)
        rest_leaf = Leaf(token.STRING, rest_value)
        insert_str_child(rest_leaf)
        self._maybe_normalize_string_quotes(rest_leaf)
        last_line = line.clone()
        maybe_append_string_operators(last_line)
        if is_valid_index(string_idx + 1):
            temp_value = rest_value
            for leaf in LL[string_idx + 1:]:
                temp_value += str(leaf)
                if leaf.type == token.LPAR:
                    break
            if str_width(temp_value) <= max_last_string_column() or LL[string_idx + 1].type == token.COMMA:
                last_line.append(rest_leaf)
                append_leaves(last_line, line, LL[string_idx + 1:])
                yield Ok(last_line)
            else:
                last_line.append(rest_leaf)
                yield Ok(last_line)
                non_string_line = line.clone()
                append_leaves(non_string_line, line, LL[string_idx + 1:])
                yield Ok(non_string_line)
        else:
            last_line.append(rest_leaf)
            last_line.comments = line.comments.copy()
            yield Ok(last_line)

    def _iter_nameescape_slices(self, string: str) -> Iterator[Tuple[Index, Index]]:
        """
        Yields:
            All ranges of @string which, if @string were to be split there,
            would result in the splitting of an \\N{...} expression (which is NOT
            allowed).
        """
        previous_was_unescaped_backslash = False
        it = iter(enumerate(string))
        for idx, c in it:
            if c == '\\':
                previous_was_unescaped_backslash = not previous_was_unescaped_backslash
                continue
            if not previous_was_unescaped_backslash or c != 'N':
                previous_was_unescaped_backslash = False
                continue
            previous_was_unescaped_backslash = False
            begin = idx - 1
            for idx, c in it:
                if c == '}':
                    end = idx
                    break
            else:
                raise RuntimeError(f'{self.__class__.__name__} LOGIC ERROR!')
            yield (begin, end)

    def _iter_fexpr_slices(self, string: str) -> Iterator[Tuple[Index, Index]]:
        """
        Yields:
            All ranges of @string which, if @string were to be split there,
            would result in the splitting of an f-expression (which is NOT
            allowed).
        """
        if 'f' not in get_string_prefix(string).lower():
            return
        yield from iter_fexpr_spans(string)

    def _get_illegal_split_indices(self, string: str) -> Set[Index]:
        illegal_indices: Set[Index] = set()
        iterators = [self._iter_fexpr_slices(string), self._iter_nameescape_slices(string)]
        for it in iterators:
            for begin, end in it:
                illegal_indices.update(range(begin, end + 1))
        return illegal_indices

    def _get_break_idx(self, string: str, max_break_idx: int) -> Optional[int]:
        """
        This method contains the algorithm that StringSplitter uses to
        determine which character to split each string at.

        Args:
            @string: The substring that we are attempting to split.
            @max_break_idx: The ideal break index. We will return this value if it
            meets all the necessary conditions. In the likely event that it
            doesn't we will try to find the closest index BELOW @max_break_idx
            that does. If that fails, we will expand our search by also
            considering all valid indices ABOVE @max_break_idx.

        Pre-Conditions:
            * assert_is_leaf_string(@string)
            * 0 <= @max_break_idx < len(@string)

        Returns:
            break_idx, if an index is able to be found that meets all of the
            conditions listed in the 'Transformations' section of this classes'
            docstring.
                OR
            None, otherwise.
        """
        is_valid_index = is_valid_index_factory(string)
        assert is_valid_index(max_break_idx)
        assert_is_leaf_string(string)
        _illegal_split_indices = self._get_illegal_split_indices(string)

        def breaks_unsplittable_expression(i: Index) -> bool:
            """
            Returns:
                True iff returning @i would result in the splitting of an
                unsplittable expression (which is NOT allowed).
            """
            return i in _illegal_split_indices

        def passes_all_checks(i: Index) -> bool:
            """
            Returns:
                True iff ALL of the conditions listed in the 'Transformations'
                section of this classes' docstring would be met by returning @i.
            """
            is_space = string[i] == ' '
            is_split_safe = is_valid_index(i - 1) and string[i - 1] in SPLIT_SAFE_CHARS
            is_not_escaped = True
            j = i - 1
            while is_valid_index(j) and string[j] == '\\':
                is_not_escaped = not is_not_escaped
                j -= 1
            is_big_enough = len(string[i:]) >= self.MIN_SUBSTR_SIZE and len(string[:i]) >= self.MIN_SUBSTR_SIZE
            return (is_space or is_split_safe) and is_not_escaped and is_big_enough and (not breaks_unsplittable_expression(i))
        break_idx = max_break_idx
        while is_valid_index(break_idx - 1) and (not passes_all_checks(break_idx)):
            break_idx -= 1
        if not passes_all_checks(break_idx):
            break_idx = max_break_idx + 1
            while is_valid_index(break_idx + 1) and (not passes_all_checks(break_idx)):
                break_idx += 1
            if not is_valid_index(break_idx) or not passes_all_checks(break_idx):
                return None
        return break_idx

    def _maybe_normalize_string_quotes(self, leaf: Leaf) -> None:
        if self.normalize_strings:
            leaf.value = normalize_string_quotes(leaf.value)

    def _normalize_f_string(self, string: str, prefix: str) -> str:
        """
        Pre-Conditions:
            * assert_is_leaf_string(@string)

        Returns:
            * If @string is an f-string that contains no f-expressions, we
            return a string identical to @string except that the 'f' prefix
            has been stripped and all double braces (i.e. '{{' or '}}') have
            been normalized (i.e. turned into '{' or '}').
                OR
            * Otherwise, we return @string.
        """
        assert_is_leaf_string(string)
        if 'f' in prefix and (not fstring_contains_expr(string)):
            new_prefix = prefix.replace('f', '')
            temp = string[len(prefix):]
            temp = re.sub('\\{\\{', '{', temp)
            temp = re.sub('\\}\\}', '}', temp)
            new_string = temp
            return f'{new_prefix}{new_string}'
        else:
            return string

    def _get_string_operator_leaves(self, leaves: Iterable[Leaf]) -> List[Leaf]:
        LL = list(leaves)
        string_op_leaves = []
        i = 0
        while LL[i].type in self.STRING_OPERATORS + [token.NAME]:
            prefix_leaf = Leaf(LL[i].type, str(LL[i]).strip())
            string_op_leaves.append(prefix_leaf)
            i += 1
        return string_op_leaves