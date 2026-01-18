import saml2
from saml2 import SamlBase
from saml2.xmldsig import KeyInfo
class SPCertEnc(SPCertEncType_):
    """The urn:net:eustix:names:tc:PEFIM:0.0:assertion:SPCertEnc element"""
    c_tag = 'SPCertEnc'
    c_namespace = NAMESPACE
    c_children = SPCertEncType_.c_children.copy()
    c_attributes = SPCertEncType_.c_attributes.copy()
    c_child_order = SPCertEncType_.c_child_order[:]
    c_cardinality = SPCertEncType_.c_cardinality.copy()