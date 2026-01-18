import json
import warnings
from typing import IO, Optional, Type, Union
from rdflib.graph import ConjunctiveGraph, Graph
from rdflib.namespace import RDF, XSD
from rdflib.serializer import Serializer
from rdflib.term import BNode, Literal, Node, URIRef
def _hex_line(self, triple, context):
    if isinstance(triple[0], (URIRef, BNode)):
        value = triple[2] if isinstance(triple[2], Literal) else self._iri_or_bn(triple[2])
        if isinstance(triple[2], URIRef):
            datatype = 'globalId'
        elif isinstance(triple[2], BNode):
            datatype = 'localId'
        elif isinstance(triple[2], Literal):
            if triple[2].datatype is not None:
                datatype = f'{triple[2].datatype}'
            elif triple[2].language is not None:
                datatype = RDF.langString
            else:
                datatype = XSD.string
        else:
            return None
        if isinstance(triple[2], Literal):
            if triple[2].language is not None:
                language = f'{triple[2].language}'
            else:
                language = ''
        else:
            language = ''
        return json.dumps([self._iri_or_bn(triple[0]), triple[1], value, datatype, language, self._context(context)]) + '\n'
    else:
        return None