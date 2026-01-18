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
def _get_bnodeid(self, ref: str) -> Optional[str]:
    if not ref.startswith('_:'):
        return
    bid = ref.split('_:', 1)[-1]
    return bid or None