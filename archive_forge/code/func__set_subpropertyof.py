import itertools
import logging
from rdflib.collection import Collection
from rdflib.graph import Graph
from rdflib.namespace import OWL, RDF, RDFS, XSD, Namespace, NamespaceManager
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
from rdflib.util import first
def _set_subpropertyof(self, other):
    if not other:
        return
    for subproperty in other:
        self.graph.add((self.identifier, RDFS.subPropertyOf, classOrIdentifier(subproperty)))