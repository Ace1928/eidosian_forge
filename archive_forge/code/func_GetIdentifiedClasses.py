import itertools
import logging
from rdflib.collection import Collection
from rdflib.graph import Graph
from rdflib.namespace import OWL, RDF, RDFS, XSD, Namespace, NamespaceManager
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
from rdflib.util import first
def GetIdentifiedClasses(graph):
    for c in graph.subjects(predicate=RDF.type, object=OWL.Class):
        if isinstance(c, URIRef):
            yield Class(c)