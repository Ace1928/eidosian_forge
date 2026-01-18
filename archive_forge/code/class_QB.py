from rdflib.namespace import DefinedNamespace, Namespace
from rdflib.term import URIRef
class QB(DefinedNamespace):
    """
    Vocabulary for multi-dimensional (e.g. statistical) data publishing

    This vocabulary allows multi-dimensional data, such as statistics, to be published in RDF. It is based on the
    core information model from SDMX (and thus also DDI).

    Generated from: http://purl.org/linked-data/cube#
    Date: 2020-05-26 14:20:05.485176

    """
    _fail = True
    attribute: URIRef
    codeList: URIRef
    component: URIRef
    componentAttachment: URIRef
    componentProperty: URIRef
    componentRequired: URIRef
    concept: URIRef
    dataSet: URIRef
    dimension: URIRef
    hierarchyRoot: URIRef
    measure: URIRef
    measureDimension: URIRef
    measureType: URIRef
    observation: URIRef
    observationGroup: URIRef
    order: URIRef
    parentChildProperty: URIRef
    slice: URIRef
    sliceKey: URIRef
    sliceStructure: URIRef
    structure: URIRef
    Attachable: URIRef
    AttributeProperty: URIRef
    CodedProperty: URIRef
    ComponentProperty: URIRef
    ComponentSet: URIRef
    ComponentSpecification: URIRef
    DataSet: URIRef
    DataStructureDefinition: URIRef
    DimensionProperty: URIRef
    HierarchicalCodeList: URIRef
    MeasureProperty: URIRef
    Observation: URIRef
    ObservationGroup: URIRef
    Slice: URIRef
    SliceKey: URIRef
    _NS = Namespace('http://purl.org/linked-data/cube#')