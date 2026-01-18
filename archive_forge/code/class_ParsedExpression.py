from __future__ import annotations
import keyword
import warnings
from typing import Collection, List, Mapping, Optional, Set, Tuple, Union
class ParsedExpression:
    """Structure containing information about one side of an `einops`-style pattern (e.g. 'b c (h w)')."""

    def __init__(self, expression: str, *, allow_underscore: bool=False, allow_duplicates: bool=False) -> None:
        """Parse the expression and store relevant metadata.

        Args:
            expression (str): the `einops`-pattern to parse
            allow_underscore (bool): whether to allow axis identifier names to begin with an underscore
            allow_duplicates (bool): whether to allow an identifier to appear more than once in the expression
        """
        self.has_ellipsis: bool = False
        self.has_ellipsis_parenthesized: Optional[bool] = None
        self.identifiers: Set[Union[str, AnonymousAxis]] = set()
        self.has_non_unitary_anonymous_axes: bool = False
        self.composition: List[Union[List[Union[str, AnonymousAxis]], str]] = []
        if '.' in expression:
            if '...' not in expression:
                raise ValueError('Expression may contain dots only inside ellipsis (...)')
            if str.count(expression, '...') != 1 or str.count(expression, '.') != 3:
                raise ValueError('Expression may contain dots only inside ellipsis (...); only one ellipsis for tensor ')
            expression = expression.replace('...', _ellipsis)
            self.has_ellipsis = True
        bracket_group: Optional[List[Union[str, AnonymousAxis]]] = None

        def add_axis_name(x: str) -> None:
            if x in self.identifiers:
                if not (allow_underscore and x == '_') and (not allow_duplicates):
                    raise ValueError(f"Indexing expression contains duplicate dimension '{x}'")
            if x == _ellipsis:
                self.identifiers.add(_ellipsis)
                if bracket_group is None:
                    self.composition.append(_ellipsis)
                    self.has_ellipsis_parenthesized = False
                else:
                    bracket_group.append(_ellipsis)
                    self.has_ellipsis_parenthesized = True
            else:
                is_number = str.isdecimal(x)
                if is_number and int(x) == 1:
                    if bracket_group is None:
                        self.composition.append([])
                    else:
                        pass
                    return
                is_axis_name, reason = self.check_axis_name_return_reason(x, allow_underscore=allow_underscore)
                if not (is_number or is_axis_name):
                    raise ValueError(f'Invalid axis identifier: {x}\n{reason}')
                axis_name: Union[str, AnonymousAxis] = AnonymousAxis(x) if is_number else x
                self.identifiers.add(axis_name)
                if is_number:
                    self.has_non_unitary_anonymous_axes = True
                if bracket_group is None:
                    self.composition.append([axis_name])
                else:
                    bracket_group.append(axis_name)
        current_identifier = None
        for char in expression:
            if char in '() ':
                if current_identifier is not None:
                    add_axis_name(current_identifier)
                current_identifier = None
                if char == '(':
                    if bracket_group is not None:
                        raise ValueError('Axis composition is one-level (brackets inside brackets not allowed)')
                    bracket_group = []
                elif char == ')':
                    if bracket_group is None:
                        raise ValueError('Brackets are not balanced')
                    self.composition.append(bracket_group)
                    bracket_group = None
            elif str.isalnum(char) or char in ['_', _ellipsis]:
                if current_identifier is None:
                    current_identifier = char
                else:
                    current_identifier += char
            else:
                raise ValueError(f"Unknown character '{char}'")
        if bracket_group is not None:
            raise ValueError(f"Imbalanced parentheses in expression: '{expression}'")
        if current_identifier is not None:
            add_axis_name(current_identifier)

    @staticmethod
    def check_axis_name_return_reason(name: str, allow_underscore: bool=False) -> Tuple[bool, str]:
        """Check if the given axis name is valid, and a message explaining why if not.

        Valid axes names are python identifiers except keywords, and should not start or end with an underscore.

        Args:
            name (str): the axis name to check
            allow_underscore (bool): whether axis names are allowed to start with an underscore

        Returns:
            Tuple[bool, str]: whether the axis name is valid, a message explaining why if not
        """
        if not str.isidentifier(name):
            return (False, 'not a valid python identifier')
        elif name[0] == '_' or name[-1] == '_':
            if name == '_' and allow_underscore:
                return (True, '')
            return (False, 'axis name should should not start or end with underscore')
        else:
            if keyword.iskeyword(name):
                warnings.warn(f'It is discouraged to use axes names that are keywords: {name}', RuntimeWarning)
            if name in ['axis']:
                warnings.warn("It is discouraged to use 'axis' as an axis name and will raise an error in future", FutureWarning)
            return (True, '')

    @staticmethod
    def check_axis_name(name: str) -> bool:
        """Check if the name is a valid axis name.

        Args:
            name (str): the axis name to check

        Returns:
            bool: whether the axis name is valid
        """
        is_valid, _ = ParsedExpression.check_axis_name_return_reason(name)
        return is_valid