from rdflib.namespace import DefinedNamespace, Namespace
from rdflib.term import URIRef
class SOSA(DefinedNamespace):
    """
    Sensor, Observation, Sample, and Actuator (SOSA) Ontology

    This ontology is based on the SSN Ontology by the W3C Semantic Sensor Networks Incubator Group (SSN-XG),
    together with considerations from the W3C/OGC Spatial Data on the Web Working Group.

    Generated from: http://www.w3.org/ns/sosa/
    Date: 2020-05-26 14:20:08.792504

    """
    ActuatableProperty: URIRef
    Actuation: URIRef
    Actuator: URIRef
    FeatureOfInterest: URIRef
    ObservableProperty: URIRef
    Observation: URIRef
    Platform: URIRef
    Procedure: URIRef
    Result: URIRef
    Sample: URIRef
    Sampler: URIRef
    Sampling: URIRef
    Sensor: URIRef
    hasSimpleResult: URIRef
    resultTime: URIRef
    actsOnProperty: URIRef
    hasFeatureOfInterest: URIRef
    hasResult: URIRef
    hasSample: URIRef
    hosts: URIRef
    isActedOnBy: URIRef
    isFeatureOfInterestOf: URIRef
    isHostedBy: URIRef
    isObservedBy: URIRef
    isResultOf: URIRef
    isSampleOf: URIRef
    madeActuation: URIRef
    madeByActuator: URIRef
    madeBySampler: URIRef
    madeBySensor: URIRef
    madeObservation: URIRef
    madeSampling: URIRef
    observedProperty: URIRef
    observes: URIRef
    phenomenonTime: URIRef
    usedProcedure: URIRef
    _NS = Namespace('http://www.w3.org/ns/sosa/')