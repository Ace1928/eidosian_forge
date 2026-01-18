import saml2
from saml2 import SamlBase
from saml2 import xmldsig as ds
class AgreementMethod(AgreementMethodType_):
    """The http://www.w3.org/2001/04/xmlenc#:AgreementMethod element"""
    c_tag = 'AgreementMethod'
    c_namespace = NAMESPACE
    c_children = AgreementMethodType_.c_children.copy()
    c_attributes = AgreementMethodType_.c_attributes.copy()
    c_child_order = AgreementMethodType_.c_child_order[:]
    c_cardinality = AgreementMethodType_.c_cardinality.copy()