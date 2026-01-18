import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
from saml2 import xmlenc as xenc
class AdditionalMetadataLocation(AdditionalMetadataLocationType_):
    """The urn:oasis:names:tc:SAML:2.0:metadata:AdditionalMetadataLocation
    element"""
    c_tag = 'AdditionalMetadataLocation'
    c_namespace = NAMESPACE
    c_children = AdditionalMetadataLocationType_.c_children.copy()
    c_attributes = AdditionalMetadataLocationType_.c_attributes.copy()
    c_child_order = AdditionalMetadataLocationType_.c_child_order[:]
    c_cardinality = AdditionalMetadataLocationType_.c_cardinality.copy()