import itertools
import logging
from rdflib.collection import Collection
from rdflib.graph import Graph
from rdflib.namespace import OWL, RDF, RDFS, XSD, Namespace, NamespaceManager
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
from rdflib.util import first
class Infix:

    def __init__(self, function):
        self.function = function

    def __rlshift__(self, other):
        return Infix(lambda x, self=self, other=other: self.function(other, x))

    def __rshift__(self, other):
        return self.function(other)

    def __rmatmul__(self, other):
        return Infix(lambda x, self=self, other=other: self.function(other, x))

    def __matmul__(self, other):
        return self.function(other)

    def __call__(self, value1, value2):
        return self.function(value1, value2)