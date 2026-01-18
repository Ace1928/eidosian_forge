import itertools
import logging
from rdflib.collection import Collection
from rdflib.graph import Graph
from rdflib.namespace import OWL, RDF, RDFS, XSD, Namespace, NamespaceManager
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
from rdflib.util import first
class MalformedClass(ValueError):
    """
    .. deprecated:: TODO-NEXT-VERSION
       This class will be removed in version ``7.0.0``.
    """
    pass