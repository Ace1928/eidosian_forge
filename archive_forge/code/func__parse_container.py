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
def _parse_container(self, context: Context, term: Term, obj: Dict[str, Any]) -> List[Any]:
    if LANG in term.container:
        obj_nodes = []
        for lang, values in obj.items():
            if not isinstance(values, list):
                values = [values]
            if lang in context.get_keys(NONE):
                obj_nodes += values
            else:
                for v in values:
                    obj_nodes.append((v, lang))
        return obj_nodes
    v11 = context.version >= 1.1
    if v11 and GRAPH in term.container and (ID in term.container):
        return [dict({GRAPH: o}) if k in context.get_keys(NONE) else dict({ID: k, GRAPH: o}) if isinstance(o, dict) else o for k, o in obj.items()]
    elif v11 and GRAPH in term.container and (INDEX in term.container):
        return [dict({GRAPH: o}) for k, o in obj.items()]
    elif v11 and GRAPH in term.container:
        return [dict({GRAPH: obj})]
    elif v11 and ID in term.container:
        return [dict({ID: k}, **o) if isinstance(o, dict) and k not in context.get_keys(NONE) else o for k, o in obj.items()]
    elif v11 and TYPE in term.container:
        return [self._add_type(context, {ID: context.expand(o) if term.type == VOCAB else o} if isinstance(o, str) else o, k) if isinstance(o, (dict, str)) and k not in context.get_keys(NONE) else o for k, o in obj.items()]
    elif INDEX in term.container:
        obj_nodes = []
        for key, nodes in obj.items():
            if not isinstance(nodes, list):
                nodes = [nodes]
            for node in nodes:
                if v11 and term.index and (key not in context.get_keys(NONE)):
                    if not isinstance(node, dict):
                        node = {ID: node}
                    values = node.get(term.index, [])
                    if not isinstance(values, list):
                        values = [values]
                    values.append(key)
                    node[term.index] = values
                obj_nodes.append(node)
        return obj_nodes
    return [obj]