from rdflib.namespace import DefinedNamespace, Namespace
from rdflib.term import URIRef
class VOID(DefinedNamespace):
    """
    Vocabulary of Interlinked Datasets (VoID)

    The Vocabulary of Interlinked Datasets (VoID) is an RDF Schema vocabulary for expressing metadata about RDF
    datasets. It is intended as a bridge between the publishers and users of RDF data, with applications ranging
    from data discovery to cataloging and archiving of datasets. This document provides a formal definition of the
    new RDF classes and properties introduced for VoID. It is a companion to the main specification document for
    VoID, <em><a href="http://www.w3.org/TR/void/">Describing Linked Datasets with the VoID Vocabulary</a></em>.

    Generated from: http://rdfs.org/ns/void#
    Date: 2020-05-26 14:20:11.911298

    """
    _fail = True
    classPartition: URIRef
    classes: URIRef
    dataDump: URIRef
    distinctObjects: URIRef
    distinctSubjects: URIRef
    documents: URIRef
    entities: URIRef
    exampleResource: URIRef
    feature: URIRef
    inDataset: URIRef
    linkPredicate: URIRef
    objectsTarget: URIRef
    openSearchDescription: URIRef
    properties: URIRef
    property: URIRef
    propertyPartition: URIRef
    rootResource: URIRef
    sparqlEndpoint: URIRef
    subjectsTarget: URIRef
    subset: URIRef
    target: URIRef
    triples: URIRef
    uriLookupEndpoint: URIRef
    uriRegexPattern: URIRef
    uriSpace: URIRef
    vocabulary: URIRef
    Dataset: URIRef
    DatasetDescription: URIRef
    Linkset: URIRef
    TechnicalFeature: URIRef
    _extras = ['class']
    _NS = Namespace('http://rdfs.org/ns/void#')