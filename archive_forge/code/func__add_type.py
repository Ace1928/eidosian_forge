from __future__ import annotations
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
import rdflib.parser
from rdflib.graph import ConjunctiveGraph, Graph
from rdflib.namespace import RDF, XSD
from rdflib.parser import InputSource, URLInputSource
from rdflib.term import BNode, IdentifiedNode, Literal, Node, URIRef
from ..shared.jsonld.context import UNDEF, Context, Term
from ..shared.jsonld.keys import (
from ..shared.jsonld.util import (
@staticmethod
def _add_type(context: Context, o: Dict[str, Any], k: str) -> Dict[str, Any]:
    otype = context.get_type(o) or []
    if otype and (not isinstance(otype, list)):
        otype = [otype]
    otype.append(k)
    o[TYPE] = otype
    return o