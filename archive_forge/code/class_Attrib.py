import sys
import re
import operator
import typing
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, Union
class Attrib:
    """
    Represents selector[namespace|attrib operator value]
    """

    @typing.overload
    def __init__(self, selector: Tree, namespace: Optional[str], attrib: str, operator: 'typing.Literal["exists"]', value: None) -> None:
        ...

    @typing.overload
    def __init__(self, selector: Tree, namespace: Optional[str], attrib: str, operator: str, value: 'Token') -> None:
        ...

    def __init__(self, selector: Tree, namespace: Optional[str], attrib: str, operator: str, value: Optional['Token']) -> None:
        self.selector = selector
        self.namespace = namespace
        self.attrib = attrib
        self.operator = operator
        self.value = value

    def __repr__(self) -> str:
        if self.namespace:
            attrib = '%s|%s' % (self.namespace, self.attrib)
        else:
            attrib = self.attrib
        if self.operator == 'exists':
            return '%s[%r[%s]]' % (self.__class__.__name__, self.selector, attrib)
        else:
            return '%s[%r[%s %s %r]]' % (self.__class__.__name__, self.selector, attrib, self.operator, typing.cast('Token', self.value).value)

    def canonical(self) -> str:
        if self.namespace:
            attrib = '%s|%s' % (self.namespace, self.attrib)
        else:
            attrib = self.attrib
        if self.operator == 'exists':
            op = attrib
        else:
            op = '%s%s%s' % (attrib, self.operator, typing.cast('Token', self.value).css())
        return '%s[%s]' % (self.selector.canonical(), op)

    def specificity(self) -> Tuple[int, int, int]:
        a, b, c = self.selector.specificity()
        b += 1
        return (a, b, c)