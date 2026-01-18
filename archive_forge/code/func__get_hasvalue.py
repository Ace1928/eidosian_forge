import itertools
import logging
from rdflib.collection import Collection
from rdflib.graph import Graph
from rdflib.namespace import OWL, RDF, RDFS, XSD, Namespace, NamespaceManager
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
from rdflib.util import first
def _get_hasvalue(self):
    for i in self.graph.objects(subject=self.identifier, predicate=OWL.hasValue):
        return Class(i, graph=self.graph)
    return None