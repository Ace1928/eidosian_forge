import itertools
import logging
from rdflib.collection import Collection
from rdflib.graph import Graph
from rdflib.namespace import OWL, RDF, RDFS, XSD, Namespace, NamespaceManager
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
from rdflib.util import first
def _get_subclassof(self):
    for anc in self.graph.objects(subject=self.identifier, predicate=RDFS.subClassOf):
        yield Class(anc, graph=self.graph, skipOWLClassMembership=True)