import sys
import re
import operator
import typing
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, Union
class Matching:
    """
    Represents selector:is(selector_list)
    """

    def __init__(self, selector: Tree, selector_list: Iterable[Tree]):
        self.selector = selector
        self.selector_list = selector_list

    def __repr__(self) -> str:
        return '%s[%r:is(%s)]' % (self.__class__.__name__, self.selector, ', '.join(map(repr, self.selector_list)))

    def canonical(self) -> str:
        selector_arguments = []
        for s in self.selector_list:
            selarg = s.canonical()
            selector_arguments.append(selarg.lstrip('*'))
        return '%s:is(%s)' % (self.selector.canonical(), ', '.join(map(str, selector_arguments)))

    def specificity(self) -> Tuple[int, int, int]:
        return max((x.specificity() for x in self.selector_list))