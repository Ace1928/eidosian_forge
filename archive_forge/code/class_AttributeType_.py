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
class AttributeType_(SamlBase):
    """The urn:oasis:names:tc:SAML:2.0:assertion:AttributeType element"""
    c_tag = 'AttributeType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{urn:oasis:names:tc:SAML:2.0:assertion}AttributeValue'] = ('attribute_value', [AttributeValue])
    c_cardinality['attribute_value'] = {'min': 0}
    c_attributes['Name'] = ('name', 'string', True)
    c_attributes['NameFormat'] = ('name_format', 'anyURI', False)
    c_attributes['FriendlyName'] = ('friendly_name', 'string', False)
    c_child_order.extend(['attribute_value'])
    c_any_attribute = {'namespace': '##other', 'processContents': 'lax'}

    def __init__(self, attribute_value=None, name=None, name_format=NAME_FORMAT_URI, friendly_name=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.attribute_value = attribute_value or []
        self.name = name
        self.name_format = name_format
        self.friendly_name = friendly_name

    def harvest_element_tree(self, tree):
        tree.attrib.setdefault('NameFormat', NAME_FORMAT_UNSPECIFIED)
        SamlBase.harvest_element_tree(self, tree)