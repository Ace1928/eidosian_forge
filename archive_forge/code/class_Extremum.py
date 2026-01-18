from __future__ import annotations
from decimal import Decimal
from typing import (
from rdflib.namespace import XSD
from rdflib.plugins.sparql.datatypes import type_promotion
from rdflib.plugins.sparql.evalutils import _eval, _val
from rdflib.plugins.sparql.operators import numeric
from rdflib.plugins.sparql.parserutils import CompValue
from rdflib.plugins.sparql.sparql import FrozenBindings, NotBoundError, SPARQLTypeError
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
class Extremum(Accumulator):
    """abstract base class for Minimum and Maximum"""

    def __init__(self, aggregation: CompValue):
        self.compare: Callable[[Any, Any], Any]
        super(Extremum, self).__init__(aggregation)
        self.value: Any = None
        self.use_row = self.dont_care

    def set_value(self, bindings: MutableMapping[Variable, Identifier]) -> None:
        if self.value is not None:
            bindings[self.var] = Literal(self.value)

    def update(self, row: FrozenBindings, aggregator: 'Aggregator') -> None:
        try:
            if self.value is None:
                self.value = _eval(self.expr, row)
            else:
                self.value = self.compare(self.value, _eval(self.expr, row))
        except NotBoundError:
            pass
        except SPARQLTypeError:
            pass