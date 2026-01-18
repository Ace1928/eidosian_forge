import itertools
import logging
from rdflib.collection import Collection
from rdflib.graph import Graph
from rdflib.namespace import OWL, RDF, RDFS, XSD, Namespace, NamespaceManager
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
from rdflib.util import first
def _get_extent(self, graph=None):
    for triple in (graph is None and self.graph or graph).triples((None, self.identifier, None)):
        yield triple