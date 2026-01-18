import itertools
import logging
from rdflib.collection import Collection
from rdflib.graph import Graph
from rdflib.namespace import OWL, RDF, RDFS, XSD, Namespace, NamespaceManager
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
from rdflib.util import first
def _set_sameAs(self, term):
    if isinstance(term, (Individual, Identifier)):
        self.graph.add((self.identifier, OWL.sameAs, classOrIdentifier(term)))
    else:
        for c in term:
            assert isinstance(c, (Individual, Identifier))
            self.graph.add((self.identifier, OWL.sameAs, classOrIdentifier(c)))