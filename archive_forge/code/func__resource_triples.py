from rdflib.namespace import RDF
from rdflib.paths import Path
from rdflib.term import BNode, Node, URIRef
def _resource_triples(self, triples):
    for s, p, o in triples:
        yield (self._cast(s), self._cast(p), self._cast(o))