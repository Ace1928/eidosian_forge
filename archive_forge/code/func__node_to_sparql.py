import collections
import re
from typing import (
from rdflib.graph import DATASET_DEFAULT_GRAPH_ID, Graph
from rdflib.plugins.stores.regexmatching import NATIVE_REGEX
from rdflib.store import Store
from rdflib.term import BNode, Identifier, Node, URIRef, Variable
from .sparqlconnector import SPARQLConnector
def _node_to_sparql(node: 'Node') -> str:
    if isinstance(node, BNode):
        raise Exception('SPARQLStore does not support BNodes! See http://www.w3.org/TR/sparql11-query/#BGPsparqlBNodes')
    return node.n3()