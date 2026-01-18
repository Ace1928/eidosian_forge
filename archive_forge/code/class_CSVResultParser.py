from __future__ import annotations
import codecs
import csv
from typing import IO, Dict, List, Optional, Union
from rdflib.plugins.sparql.processor import SPARQLResult
from rdflib.query import Result, ResultParser, ResultSerializer
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
class CSVResultParser(ResultParser):

    def __init__(self):
        self.delim = ','

    def parse(self, source: IO, content_type: Optional[str]=None) -> Result:
        r = Result('SELECT')
        if isinstance(source.read(0), bytes):
            source = codecs.getreader('utf-8')(source)
        reader = csv.reader(source, delimiter=self.delim)
        r.vars = [Variable(x) for x in next(reader)]
        r.bindings = []
        for row in reader:
            r.bindings.append(self.parseRow(row, r.vars))
        return r

    def parseRow(self, row: List[str], v: List[Variable]) -> Dict[Variable, Union[BNode, URIRef, Literal]]:
        return dict(((var, val) for var, val in zip(v, [self.convertTerm(t) for t in row]) if val is not None))

    def convertTerm(self, t: str) -> Optional[Union[BNode, URIRef, Literal]]:
        if t == '':
            return None
        if t.startswith('_:'):
            return BNode(t)
        if t.startswith('http://') or t.startswith('https://'):
            return URIRef(t)
        return Literal(t)