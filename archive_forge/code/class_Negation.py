import sys
import re
import operator
import typing
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, Union
class Negation:
    """
    Represents selector:not(subselector)
    """

    def __init__(self, selector: Tree, subselector: Tree) -> None:
        self.selector = selector
        self.subselector = subselector

    def __repr__(self) -> str:
        return '%s[%r:not(%r)]' % (self.__class__.__name__, self.selector, self.subselector)

    def canonical(self) -> str:
        subsel = self.subselector.canonical()
        if len(subsel) > 1:
            subsel = subsel.lstrip('*')
        return '%s:not(%s)' % (self.selector.canonical(), subsel)

    def specificity(self) -> Tuple[int, int, int]:
        a1, b1, c1 = self.selector.specificity()
        a2, b2, c2 = self.subselector.specificity()
        return (a1 + a2, b1 + b2, c1 + c2)