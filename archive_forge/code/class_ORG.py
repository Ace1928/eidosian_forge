from rdflib.namespace import DefinedNamespace, Namespace
from rdflib.term import URIRef
class ORG(DefinedNamespace):
    """
    Core organization ontology

    Vocabulary for describing organizational structures, specializable to a broad variety of types of
    organization.

    Generated from: http://www.w3.org/ns/org#
    Date: 2020-05-26 14:20:02.908408

    """
    _fail = True
    basedAt: URIRef
    changedBy: URIRef
    classification: URIRef
    hasMember: URIRef
    hasMembership: URIRef
    hasPost: URIRef
    hasPrimarySite: URIRef
    hasRegisteredSite: URIRef
    hasSite: URIRef
    hasSubOrganization: URIRef
    hasUnit: URIRef
    headOf: URIRef
    heldBy: URIRef
    holds: URIRef
    identifier: URIRef
    linkedTo: URIRef
    location: URIRef
    member: URIRef
    memberDuring: URIRef
    memberOf: URIRef
    organization: URIRef
    originalOrganization: URIRef
    postIn: URIRef
    purpose: URIRef
    remuneration: URIRef
    reportsTo: URIRef
    resultedFrom: URIRef
    resultingOrganization: URIRef
    role: URIRef
    roleProperty: URIRef
    siteAddress: URIRef
    siteOf: URIRef
    subOrganizationOf: URIRef
    transitiveSubOrganizationOf: URIRef
    unitOf: URIRef
    ChangeEvent: URIRef
    FormalOrganization: URIRef
    Membership: URIRef
    Organization: URIRef
    OrganizationalCollaboration: URIRef
    OrganizationalUnit: URIRef
    Post: URIRef
    Role: URIRef
    Site: URIRef
    Head: URIRef
    _NS = Namespace('http://www.w3.org/ns/org#')