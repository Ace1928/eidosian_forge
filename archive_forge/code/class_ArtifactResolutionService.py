import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
from saml2 import xmlenc as xenc
class ArtifactResolutionService(IndexedEndpointType_):
    """The urn:oasis:names:tc:SAML:2.0:metadata:ArtifactResolutionService
    element"""
    c_tag = 'ArtifactResolutionService'
    c_namespace = NAMESPACE
    c_children = IndexedEndpointType_.c_children.copy()
    c_attributes = IndexedEndpointType_.c_attributes.copy()
    c_child_order = IndexedEndpointType_.c_child_order[:]
    c_cardinality = IndexedEndpointType_.c_cardinality.copy()