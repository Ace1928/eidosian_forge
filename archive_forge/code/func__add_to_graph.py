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
def _add_to_graph(self, dataset: Graph, graph: Graph, context: Context, node: Any, topcontext: bool=False) -> Optional[Node]:
    if not isinstance(node, dict) or context.get_value(node):
        return
    if CONTEXT in node and (not topcontext):
        local_context = node[CONTEXT]
        if local_context:
            context = context.subcontext(local_context)
        else:
            context = Context(base=context.doc_base)
    context = context.get_context_for_type(node)
    id_val = context.get_id(node)
    if id_val is None:
        nested_id = self._get_nested_id(context, node)
        if nested_id is not None and len(nested_id) > 0:
            id_val = nested_id
    if isinstance(id_val, str):
        subj = self._to_rdf_id(context, id_val)
    else:
        subj = BNode()
    if subj is None:
        return None
    no_id = id_val is None
    for key, obj in node.items():
        if key == CONTEXT or key in context.get_keys(ID):
            continue
        if key == REV or key in context.get_keys(REV):
            for rkey, robj in obj.items():
                self._key_to_graph(dataset, graph, context, subj, rkey, robj, reverse=True, no_id=no_id)
        else:
            self._key_to_graph(dataset, graph, context, subj, key, obj, no_id=no_id)
    return subj