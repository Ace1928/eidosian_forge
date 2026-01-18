import re
import typing
import warnings
from typing import Optional
from cssselect.parser import (
class XPathExpr:

    def __init__(self, path: str='', element: str='*', condition: str='', star_prefix: bool=False) -> None:
        self.path = path
        self.element = element
        self.condition = condition

    def __str__(self) -> str:
        path = str(self.path) + str(self.element)
        if self.condition:
            path += '[%s]' % self.condition
        return path

    def __repr__(self) -> str:
        return '%s[%s]' % (self.__class__.__name__, self)

    def add_condition(self, condition: str, conjuction: str='and') -> 'XPathExpr':
        if self.condition:
            self.condition = '(%s) %s (%s)' % (self.condition, conjuction, condition)
        else:
            self.condition = condition
        return self

    def add_name_test(self) -> None:
        if self.element == '*':
            return
        self.add_condition('name() = %s' % GenericTranslator.xpath_literal(self.element))
        self.element = '*'

    def add_star_prefix(self) -> None:
        """
        Append '*/' to the path to keep the context constrained
        to a single parent.
        """
        self.path += '*/'

    def join(self, combiner: str, other: 'XPathExpr', closing_combiner: Optional[str]=None, has_inner_condition: bool=False) -> 'XPathExpr':
        path = str(self) + combiner
        if other.path != '*/':
            path += other.path
        self.path = path
        if not has_inner_condition:
            self.element = other.element + closing_combiner if closing_combiner else other.element
            self.condition = other.condition
        else:
            self.element = other.element
            if other.condition:
                self.element += '[' + other.condition + ']'
            if closing_combiner:
                self.element += closing_combiner
        return self