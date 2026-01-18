from __future__ import annotations
import re
from typing import (
import warnings
from pandas.errors import CSSWarning
from pandas.util._exceptions import find_stack_level
def _border_expander(side: str='') -> Callable:
    """
    Wrapper to expand 'border' property into border color, style, and width properties

    Parameters
    ----------
    side : str
        The border side to expand into properties

    Returns
    -------
        function: Return to call when a 'border(-{side}): {value}' string is encountered
    """
    if side != '':
        side = f'-{side}'

    def expand(self, prop, value: str) -> Generator[tuple[str, str], None, None]:
        """
        Expand border into color, style, and width tuples

        Parameters
        ----------
            prop : str
                CSS property name passed to styler
            value : str
                Value passed to styler for property

        Yields
        ------
            Tuple (str, str): Expanded property, value
        """
        tokens = value.split()
        if len(tokens) == 0 or len(tokens) > 3:
            warnings.warn(f'Too many tokens provided to "{prop}" (expected 1-3)', CSSWarning, stacklevel=find_stack_level())
        border_declarations = {f'border{side}-color': 'black', f'border{side}-style': 'none', f'border{side}-width': 'medium'}
        for token in tokens:
            if token.lower() in self.BORDER_STYLES:
                border_declarations[f'border{side}-style'] = token
            elif any((ratio in token.lower() for ratio in self.BORDER_WIDTH_RATIOS)):
                border_declarations[f'border{side}-width'] = token
            else:
                border_declarations[f'border{side}-color'] = token
        yield from self.atomize(border_declarations.items())
    return expand