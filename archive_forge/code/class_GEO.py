from rdflib.namespace import DefinedNamespace, Namespace
from rdflib.term import URIRef
class GEO(DefinedNamespace):
    """
    An RDF/OWL vocabulary for representing spatial information

    Generated from: http://schemas.opengis.net/geosparql/1.0/geosparql_vocab_all.rdf
    Date: 2021-12-27 17:38:15.101187

    .. code-block:: Turtle

        <http://www.opengis.net/ont/geosparql> dc:creator "Open Geospatial Consortium"^^xsd:string
        dc:date "2012-04-30"^^xsd:date
        dc:source <http://www.opengis.net/doc/IS/geosparql/1.0>
            "OGC GeoSPARQL – A Geographic Query Language for RDF Data OGC 11-052r5"^^xsd:string
        rdfs:seeAlso <http://www.opengis.net/def/function/ogc-geosparql/1.0>
            <http://www.opengis.net/def/rule/ogc-geosparql/1.0>
            <http://www.opengis.net/doc/IS/geosparql/1.0>
        owl:imports dc:
            <http://www.opengis.net/ont/gml>
            <http://www.opengis.net/ont/sf>
            <http://www.w3.org/2004/02/skos/core>
        owl:versionInfo "OGC GeoSPARQL 1.0"^^xsd:string
    """
    gmlLiteral: URIRef
    wktLiteral: URIRef
    Feature: URIRef
    Geometry: URIRef
    SpatialObject: URIRef
    asGML: URIRef
    asWKT: URIRef
    coordinateDimension: URIRef
    dimension: URIRef
    hasSerialization: URIRef
    isEmpty: URIRef
    isSimple: URIRef
    spatialDimension: URIRef
    defaultGeometry: URIRef
    ehContains: URIRef
    ehCoveredBy: URIRef
    ehCovers: URIRef
    ehDisjoint: URIRef
    ehEquals: URIRef
    ehInside: URIRef
    ehMeet: URIRef
    ehOverlap: URIRef
    hasGeometry: URIRef
    rcc8dc: URIRef
    rcc8ec: URIRef
    rcc8eq: URIRef
    rcc8ntpp: URIRef
    rcc8ntppi: URIRef
    rcc8po: URIRef
    rcc8tpp: URIRef
    rcc8tppi: URIRef
    sfContains: URIRef
    sfCrosses: URIRef
    sfDisjoint: URIRef
    sfEquals: URIRef
    sfIntersects: URIRef
    sfOverlaps: URIRef
    sfTouches: URIRef
    sfWithin: URIRef
    _NS = Namespace('http://www.opengis.net/ont/geosparql#')