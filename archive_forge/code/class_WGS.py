from rdflib.namespace import DefinedNamespace, Namespace
from rdflib.term import URIRef
class WGS(DefinedNamespace):
    """
    Basic Geo (WGS84 lat/long) Vocabulary

    The HTML Specification for the vocabulary can be found
    `here <https://www.w3.org/2003/01/geo/>`.
    """
    _NS = Namespace('https://www.w3.org/2003/01/geo/wgs84_pos#')
    SpatialThing: URIRef
    Point: URIRef
    alt: URIRef
    lat: URIRef
    lat_long: URIRef
    location: URIRef
    long: URIRef