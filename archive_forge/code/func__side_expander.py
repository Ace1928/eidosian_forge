from __future__ import annotations
import re
from typing import (
import warnings
from pandas.errors import CSSWarning
from pandas.util._exceptions import find_stack_level
def _side_expander(prop_fmt: str) -> Callable:
    """
    Wrapper to expand shorthand property into top, right, bottom, left properties

    Parameters
    ----------
    side : str
        The border side to expand into properties

    Returns
    -------
        function: Return to call when a 'border(-{side}): {value}' string is encountered
    """

    def expand(self, prop, value: str) -> Generator[tuple[str, str], None, None]:
        """
        Expand shorthand property into side-specific property (top, right, bottom, left)

        Parameters
        ----------
            prop (str): CSS property name
            value (str): String token for property

        Yields
        ------
            Tuple (str, str): Expanded property, value
        """
        tokens = value.split()
        try:
            mapping = self.SIDE_SHORTHANDS[len(tokens)]
        except KeyError:
            warnings.warn(f'Could not expand "{prop}: {value}"', CSSWarning, stacklevel=find_stack_level())
            return
        for key, idx in zip(self.SIDES, mapping):
            yield (prop_fmt.format(key), tokens[idx])
    return expand