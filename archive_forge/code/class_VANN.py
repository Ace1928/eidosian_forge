from rdflib.namespace import DefinedNamespace, Namespace
from rdflib.term import URIRef
class VANN(DefinedNamespace):
    """
    VANN: A vocabulary for annotating vocabulary descriptions

    This document describes a vocabulary for annotating descriptions of vocabularies with examples and usage
    notes.

    Generated from: https://vocab.org/vann/vann-vocab-20100607.rdf
    Date: 2020-05-26 14:21:15.580430

    """
    _fail = True
    changes: URIRef
    example: URIRef
    preferredNamespacePrefix: URIRef
    preferredNamespaceUri: URIRef
    termGroup: URIRef
    usageNote: URIRef
    _NS = Namespace('http://purl.org/vocab/vann/')