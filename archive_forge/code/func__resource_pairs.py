from rdflib.namespace import RDF
from rdflib.paths import Path
from rdflib.term import BNode, Node, URIRef
def _resource_pairs(self, pairs):
    for s1, s2 in pairs:
        yield (self._cast(s1), self._cast(s2))