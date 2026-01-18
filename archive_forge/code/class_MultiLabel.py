import sys
import re
from abc import ABCMeta
from unicodedata import name as unicode_name
from decimal import Decimal, DecimalException
from typing import Any, cast, overload, Callable, Dict, Generic, List, \
class MultiLabel:
    """
    Helper class for defining multi-value label for tokens. Useful when a symbol
    has more roles. A label of this type has equivalence with each of its values.

    Example:
        label = MultiLabel('function', 'operator')
        label == 'symbol'    # False
        label == 'function'  # True
        label == 'operator'  # True
    """

    def __init__(self, *values: str) -> None:
        self.values = values

    def __eq__(self, other: object) -> bool:
        return any((other == v for v in self.values))

    def __ne__(self, other: object) -> bool:
        return all((other != v for v in self.values))

    def __repr__(self) -> str:
        return '%s%s' % (self.__class__.__name__, self.values)

    def __str__(self) -> str:
        return '__'.join(self.values).replace(' ', '_')

    def __hash__(self) -> int:
        return hash(self.values)

    def __contains__(self, item: str) -> bool:
        return any((item in v for v in self.values))

    def startswith(self, s: str) -> bool:
        return any((v.startswith(s) for v in self.values))

    def endswith(self, s: str) -> bool:
        return any((v.endswith(s) for v in self.values))