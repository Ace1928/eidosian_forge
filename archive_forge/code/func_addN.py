import collections
import re
from typing import (
from rdflib.graph import DATASET_DEFAULT_GRAPH_ID, Graph
from rdflib.plugins.stores.regexmatching import NATIVE_REGEX
from rdflib.store import Store
from rdflib.term import BNode, Identifier, Node, URIRef, Variable
from .sparqlconnector import SPARQLConnector
def addN(self, quads: Iterable['_QuadType']) -> None:
    """Add a list of quads to the store."""
    if not self.update_endpoint:
        raise Exception("UpdateEndpoint is not set - call 'open'")
    contexts = collections.defaultdict(list)
    for subject, predicate, obj, context in quads:
        contexts[context].append((subject, predicate, obj))
    data: List[str] = []
    nts = self.node_to_sparql
    for context in contexts:
        triples = ['%s %s %s .' % (nts(subject), nts(predicate), nts(obj)) for subject, predicate, obj in contexts[context]]
        data.append('INSERT DATA { GRAPH %s { %s } }\n' % (nts(context.identifier), '\n'.join(triples)))
    self._transaction().extend(data)
    if self.autocommit:
        self.commit()