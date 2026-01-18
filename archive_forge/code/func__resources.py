from rdflib.namespace import RDF
from rdflib.paths import Path
from rdflib.term import BNode, Node, URIRef
def _resources(self, nodes):
    for node in nodes:
        yield self._cast(node)