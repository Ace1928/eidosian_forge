from __future__ import annotations
import json
from typing import IO, Any, Dict, Mapping, MutableSequence, Optional
from rdflib.query import Result, ResultException, ResultParser, ResultSerializer
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
class JSONResultParser(ResultParser):

    def parse(self, source: IO, content_type: Optional[str]=None) -> Result:
        inp = source.read()
        if isinstance(inp, bytes):
            inp = inp.decode('utf-8')
        return JSONResult(json.loads(inp))