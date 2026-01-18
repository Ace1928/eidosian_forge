import saml2
from saml2 import SamlBase
class GoverningAgreementRef(GoverningAgreementRefType_):
    """The urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword:GoverningAgreementRef element"""
    c_tag = 'GoverningAgreementRef'
    c_namespace = NAMESPACE
    c_children = GoverningAgreementRefType_.c_children.copy()
    c_attributes = GoverningAgreementRefType_.c_attributes.copy()
    c_child_order = GoverningAgreementRefType_.c_child_order[:]
    c_cardinality = GoverningAgreementRefType_.c_cardinality.copy()