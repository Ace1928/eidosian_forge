import base64
from datetime import date
from datetime import datetime
import saml2
from saml2 import SamlBase
from saml2 import xmldsig as ds
from saml2 import xmlenc as xenc
from saml2.validate import MustValueError
from saml2.validate import ShouldValueError
from saml2.validate import valid_domain_name
from saml2.validate import valid_ipv4
from saml2.validate import valid_ipv6
class SubjectLocality(SubjectLocalityType_):
    """The urn:oasis:names:tc:SAML:2.0:assertion:SubjectLocality element"""
    c_tag = 'SubjectLocality'
    c_namespace = NAMESPACE
    c_children = SubjectLocalityType_.c_children.copy()
    c_attributes = SubjectLocalityType_.c_attributes.copy()
    c_child_order = SubjectLocalityType_.c_child_order[:]
    c_cardinality = SubjectLocalityType_.c_cardinality.copy()

    def verify(self):
        if self.address:
            if valid_ipv4(self.address) or valid_ipv6(self.address):
                pass
            else:
                raise ShouldValueError('Not an IPv4 or IPv6 address')
        elif self.dns_name:
            valid_domain_name(self.dns_name)
        return SubjectLocalityType_.verify(self)