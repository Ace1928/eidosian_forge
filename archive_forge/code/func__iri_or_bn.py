import json
import warnings
from typing import IO, Optional, Type, Union
from rdflib.graph import ConjunctiveGraph, Graph
from rdflib.namespace import RDF, XSD
from rdflib.serializer import Serializer
from rdflib.term import BNode, Literal, Node, URIRef
def _iri_or_bn(self, i_):
    if isinstance(i_, URIRef):
        return f'{i_}'
    elif isinstance(i_, BNode):
        return f'{i_.n3()}'
    else:
        return None