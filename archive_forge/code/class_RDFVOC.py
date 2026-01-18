from rdflib.namespace import RDF
from rdflib.term import URIRef
class RDFVOC(RDF):
    _underscore_num = True
    _fail = True
    RDF: URIRef
    Description: URIRef
    ID: URIRef
    about: URIRef
    parseType: URIRef
    resource: URIRef
    li: URIRef
    nodeID: URIRef
    datatype: URIRef