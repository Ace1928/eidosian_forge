from __future__ import annotations
import json
from typing import IO, Any, Dict, Mapping, MutableSequence, Optional
from rdflib.query import Result, ResultException, ResultParser, ResultSerializer
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
def _get_bindings(self) -> MutableSequence[Mapping[Variable, Identifier]]:
    ret: MutableSequence[Mapping[Variable, Identifier]] = []
    for row in self.json['results']['bindings']:
        outRow: Dict[Variable, Identifier] = {}
        for k, v in row.items():
            outRow[Variable(k)] = parseJsonTerm(v)
        ret.append(outRow)
    return ret