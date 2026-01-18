from __future__ import annotations
import json
from typing import IO, Any, Dict, Mapping, MutableSequence, Optional
from rdflib.query import Result, ResultException, ResultParser, ResultSerializer
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
def _bindingToJSON(self, b: Mapping[Variable, Identifier]) -> Dict[Variable, Any]:
    res = {}
    for var in b:
        j = termToJSON(self, b[var])
        if j is not None:
            res[var] = termToJSON(self, b[var])
    return res