import saml2
from saml2 import md
class DiscoveryResponse(md.IndexedEndpointType_):
    """The urn:oasis:names:tc:SAML:profiles:SSO:idp-discovery-protocol:
    DiscoveryResponse element"""
    c_tag = 'DiscoveryResponse'
    c_namespace = NAMESPACE
    c_children = md.IndexedEndpointType_.c_children.copy()
    c_attributes = md.IndexedEndpointType_.c_attributes.copy()
    c_child_order = md.IndexedEndpointType_.c_child_order[:]
    c_cardinality = md.IndexedEndpointType_.c_cardinality.copy()