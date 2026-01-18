import saml2
from saml2 import SamlBase
class X509DataType_(SamlBase):
    """The http://www.w3.org/2000/09/xmldsig#:X509DataType element"""
    c_tag = 'X509DataType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{http://www.w3.org/2000/09/xmldsig#}X509IssuerSerial'] = ('x509_issuer_serial', X509IssuerSerial)
    c_cardinality['x509_issuer_serial'] = {'min': 0, 'max': 1}
    c_children['{http://www.w3.org/2000/09/xmldsig#}X509SKI'] = ('x509_ski', X509SKI)
    c_cardinality['x509_ski'] = {'min': 0, 'max': 1}
    c_children['{http://www.w3.org/2000/09/xmldsig#}X509SubjectName'] = ('x509_subject_name', X509SubjectName)
    c_cardinality['x509_subject_name'] = {'min': 0, 'max': 1}
    c_children['{http://www.w3.org/2000/09/xmldsig#}X509Certificate'] = ('x509_certificate', X509Certificate)
    c_cardinality['x509_certificate'] = {'min': 0, 'max': 1}
    c_children['{http://www.w3.org/2000/09/xmldsig#}X509CRL'] = ('x509_crl', X509CRL)
    c_cardinality['x509_crl'] = {'min': 0, 'max': 1}
    c_child_order.extend(['x509_issuer_serial', 'x509_ski', 'x509_subject_name', 'x509_certificate', 'x509_crl'])

    def __init__(self, x509_issuer_serial=None, x509_ski=None, x509_subject_name=None, x509_certificate=None, x509_crl=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.x509_issuer_serial = x509_issuer_serial
        self.x509_ski = x509_ski
        self.x509_subject_name = x509_subject_name
        self.x509_certificate = x509_certificate
        self.x509_crl = x509_crl