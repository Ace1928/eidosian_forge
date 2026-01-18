from __future__ import annotations
from collections import OrderedDict
from types import MethodType
from typing import (
from pyparsing import ParserElement, ParseResults, TokenConverter, originalTextFor
from rdflib.term import BNode, Identifier, Variable
from rdflib.plugins.sparql.sparql import NotBoundError, SPARQLError  # noqa: E402
class CompValue(OrderedDict):
    """
    The result of parsing a Comp
    Any included Params are available as Dict keys
    or as attributes

    """

    def __init__(self, name: str, **values):
        OrderedDict.__init__(self)
        self.name = name
        self.update(values)

    def clone(self) -> CompValue:
        return CompValue(self.name, **self)

    def __str__(self) -> str:
        return self.name + '_' + OrderedDict.__str__(self)

    def __repr__(self) -> str:
        return self.name + '_' + dict.__repr__(self)

    def _value(self, val: _ValT, variables: bool=False, errors: bool=False) -> Union[_ValT, Any]:
        if self.ctx is not None:
            return value(self.ctx, val, variables)
        else:
            return val

    def __getitem__(self, a):
        return self._value(OrderedDict.__getitem__(self, a))

    def get(self, a, variables: bool=False, errors: bool=False):
        return self._value(OrderedDict.get(self, a, a), variables, errors)

    def __getattr__(self, a: str) -> Any:
        if a in ('_OrderedDict__root', '_OrderedDict__end'):
            raise AttributeError()
        try:
            return self[a]
        except KeyError:
            return None
    if TYPE_CHECKING:

        def __setattr__(self, __name: str, __value: Any) -> None:
            ...