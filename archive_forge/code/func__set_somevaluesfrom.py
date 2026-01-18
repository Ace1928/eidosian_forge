import itertools
import logging
from rdflib.collection import Collection
from rdflib.graph import Graph
from rdflib.namespace import OWL, RDF, RDFS, XSD, Namespace, NamespaceManager
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
from rdflib.util import first
def _set_somevaluesfrom(self, other):
    if not other:
        return
    triple = (self.identifier, OWL.someValuesFrom, classOrIdentifier(other))
    if triple in self.graph:
        return
    else:
        self.graph.set(triple)