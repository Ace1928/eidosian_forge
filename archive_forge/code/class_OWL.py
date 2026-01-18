from rdflib.namespace import DefinedNamespace, Namespace
from rdflib.term import URIRef
class OWL(DefinedNamespace):
    """
    The OWL 2 Schema vocabulary (OWL 2)

    This ontology partially describes the built-in classes and    properties that together form the basis of
    the RDF/XML syntax of OWL 2.    The content of this ontology is based on Tables 6.1 and 6.2    in Section 6.4
    of the OWL 2 RDF-Based Semantics specification,    available at http://www.w3.org/TR/owl2-rdf-based-
    semantics/.    Please note that those tables do not include the different annotations    (labels, comments and
    rdfs:isDefinedBy links) used in this file.    Also note that the descriptions provided in this ontology do not
    provide a complete and correct formal description of either the syntax    or the semantics of the introduced
    terms (please see the OWL 2    recommendations for the complete and normative specifications).    Furthermore,
    the information provided by this ontology may be    misleading if not used with care. This ontology SHOULD NOT
    be imported    into OWL ontologies. Importing this file into an OWL 2 DL ontology    will cause it to become
    an OWL 2 Full ontology and may have other,    unexpected, consequences.

    Generated from: http://www.w3.org/2002/07/owl#
    Date: 2020-05-26 14:20:03.193795

    """
    _fail = True
    allValuesFrom: URIRef
    annotatedProperty: URIRef
    annotatedSource: URIRef
    annotatedTarget: URIRef
    assertionProperty: URIRef
    cardinality: URIRef
    complementOf: URIRef
    datatypeComplementOf: URIRef
    differentFrom: URIRef
    disjointUnionOf: URIRef
    disjointWith: URIRef
    distinctMembers: URIRef
    equivalentClass: URIRef
    equivalentProperty: URIRef
    hasKey: URIRef
    hasSelf: URIRef
    hasValue: URIRef
    intersectionOf: URIRef
    inverseOf: URIRef
    maxCardinality: URIRef
    maxQualifiedCardinality: URIRef
    members: URIRef
    minCardinality: URIRef
    minQualifiedCardinality: URIRef
    onClass: URIRef
    onDataRange: URIRef
    onDatatype: URIRef
    onProperties: URIRef
    onProperty: URIRef
    oneOf: URIRef
    propertyChainAxiom: URIRef
    propertyDisjointWith: URIRef
    qualifiedCardinality: URIRef
    sameAs: URIRef
    someValuesFrom: URIRef
    sourceIndividual: URIRef
    targetIndividual: URIRef
    targetValue: URIRef
    unionOf: URIRef
    withRestrictions: URIRef
    AllDifferent: URIRef
    AllDisjointClasses: URIRef
    AllDisjointProperties: URIRef
    Annotation: URIRef
    AnnotationProperty: URIRef
    AsymmetricProperty: URIRef
    Axiom: URIRef
    Class: URIRef
    DataRange: URIRef
    DatatypeProperty: URIRef
    DeprecatedClass: URIRef
    DeprecatedProperty: URIRef
    FunctionalProperty: URIRef
    InverseFunctionalProperty: URIRef
    IrreflexiveProperty: URIRef
    NamedIndividual: URIRef
    NegativePropertyAssertion: URIRef
    ObjectProperty: URIRef
    Ontology: URIRef
    OntologyProperty: URIRef
    ReflexiveProperty: URIRef
    Restriction: URIRef
    SymmetricProperty: URIRef
    TransitiveProperty: URIRef
    backwardCompatibleWith: URIRef
    deprecated: URIRef
    incompatibleWith: URIRef
    priorVersion: URIRef
    versionInfo: URIRef
    Nothing: URIRef
    Thing: URIRef
    bottomDataProperty: URIRef
    topDataProperty: URIRef
    bottomObjectProperty: URIRef
    topObjectProperty: URIRef
    imports: URIRef
    versionIRI: URIRef
    rational: URIRef
    real: URIRef
    _NS = Namespace('http://www.w3.org/2002/07/owl#')