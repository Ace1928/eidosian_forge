from __future__ import annotations
import json
from typing import IO, Any, Dict, Mapping, MutableSequence, Optional
from rdflib.query import Result, ResultException, ResultParser, ResultSerializer
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
class JSONResultSerializer(ResultSerializer):

    def __init__(self, result: Result):
        ResultSerializer.__init__(self, result)

    def serialize(self, stream: IO, encoding: str=None) -> None:
        res: Dict[str, Any] = {}
        if self.result.type == 'ASK':
            res['head'] = {}
            res['boolean'] = self.result.askAnswer
        else:
            res['results'] = {}
            res['head'] = {}
            res['head']['vars'] = self.result.vars
            res['results']['bindings'] = [self._bindingToJSON(x) for x in self.result.bindings]
        r = json.dumps(res, allow_nan=False, ensure_ascii=False)
        if encoding is not None:
            stream.write(r.encode(encoding))
        else:
            stream.write(r)

    def _bindingToJSON(self, b: Mapping[Variable, Identifier]) -> Dict[Variable, Any]:
        res = {}
        for var in b:
            j = termToJSON(self, b[var])
            if j is not None:
                res[var] = termToJSON(self, b[var])
        return res