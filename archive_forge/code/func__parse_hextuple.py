from __future__ import annotations
import json
import warnings
from io import TextIOWrapper
from typing import Any, BinaryIO, List, Optional, TextIO, Union
from rdflib.graph import ConjunctiveGraph, Graph
from rdflib.parser import InputSource, Parser
from rdflib.term import BNode, Literal, URIRef
def _parse_hextuple(self, cg: ConjunctiveGraph, tup: List[Union[str, None]]) -> None:
    if tup[0] is None or tup[1] is None or tup[2] is None or (tup[3] is None):
        raise ValueError(f'subject, predicate, value, datatype cannot be None. Given: {tup}')
    s: Union[URIRef, BNode]
    if tup[0].startswith('_'):
        s = BNode(value=tup[0].replace('_:', ''))
    else:
        s = URIRef(tup[0])
    p = URIRef(tup[1])
    o: Union[URIRef, BNode, Literal]
    if tup[3] == 'globalId':
        o = URIRef(tup[2])
    elif tup[3] == 'localId':
        o = BNode(value=tup[2].replace('_:', ''))
    elif tup[4] is None:
        o = Literal(tup[2], datatype=URIRef(tup[3]))
    else:
        o = Literal(tup[2], lang=tup[4])
    if tup[5] is not None:
        c = URIRef(tup[5])
        cg.add((s, p, o, c))
    else:
        cg.add((s, p, o))