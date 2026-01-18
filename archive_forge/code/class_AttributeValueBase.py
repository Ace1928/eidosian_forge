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
class AttributeValueBase(SamlBase):

    def __init__(self, text=None, extension_elements=None, extension_attributes=None):
        self._extatt = {}
        SamlBase.__init__(self, text=None, extension_elements=extension_elements, extension_attributes=extension_attributes)
        if self._extatt:
            self.extension_attributes = self._extatt
        if text:
            self.set_text(text)
        elif not extension_elements:
            self.extension_attributes = {XSI_NIL: 'true'}
        elif XSI_TYPE in self.extension_attributes:
            del self.extension_attributes[XSI_TYPE]

    def __setattr__(self, key, value):
        if key == 'text':
            self.set_text(value)
        else:
            SamlBase.__setattr__(self, key, value)

    def verify(self):
        if not self.text and (not self.extension_elements):
            if not self.extension_attributes:
                raise Exception('Attribute value base should not have extension attributes')
            if self.extension_attributes[XSI_NIL] != 'true':
                raise Exception('Attribute value base should not have extension attributes')
            return True
        else:
            SamlBase.verify(self)

    def set_type(self, typ):
        try:
            del self.extension_attributes[XSI_NIL]
        except (AttributeError, KeyError):
            pass
        try:
            self.extension_attributes[XSI_TYPE] = typ
        except AttributeError:
            self._extatt[XSI_TYPE] = typ
        if typ.startswith('xs:'):
            try:
                self.extension_attributes['xmlns:xs'] = XS_NAMESPACE
            except AttributeError:
                self._extatt['xmlns:xs'] = XS_NAMESPACE
        if typ.startswith('xsd:'):
            try:
                self.extension_attributes['xmlns:xsd'] = XS_NAMESPACE
            except AttributeError:
                self._extatt['xmlns:xsd'] = XS_NAMESPACE

    def get_type(self):
        try:
            return self.extension_attributes[XSI_TYPE]
        except (KeyError, AttributeError):
            try:
                return self._extatt[XSI_TYPE]
            except KeyError:
                return ''

    def clear_type(self):
        try:
            del self.extension_attributes[XSI_TYPE]
        except KeyError:
            pass
        try:
            del self._extatt[XSI_TYPE]
        except KeyError:
            pass

    def set_text(self, value, base64encode=False):

        def _wrong_type_value(xsd, value):
            msg = 'Type and value do not match: {xsd}:{type}:{value}'
            msg = msg.format(xsd=xsd, type=type(value), value=value)
            raise ValueError(msg)
        if isinstance(value, bytes):
            value = value.decode('utf-8')
        type_to_xsd = {str: 'string', int: 'integer', float: 'float', bool: 'boolean', type(None): ''}
        xsd_types_props = {'string': {'type': str, 'to_type': str, 'to_text': str}, 'integer': {'type': int, 'to_type': int, 'to_text': str}, 'short': {'type': int, 'to_type': int, 'to_text': str}, 'int': {'type': int, 'to_type': int, 'to_text': str}, 'long': {'type': int, 'to_type': int, 'to_text': str}, 'float': {'type': float, 'to_type': float, 'to_text': str}, 'double': {'type': float, 'to_type': float, 'to_text': str}, 'boolean': {'type': bool, 'to_type': lambda x: {'true': True, 'false': False}[str(x).lower()], 'to_text': lambda x: str(x).lower()}, 'date': {'type': date, 'to_type': lambda x: datetime.strptime(x, '%Y-%m-%d').date(), 'to_text': str}, 'base64Binary': {'type': str, 'to_type': str, 'to_text': lambda x: base64.encodebytes(x.encode()) if base64encode else x}, 'anyType': {'type': type(value), 'to_type': lambda x: x, 'to_text': lambda x: x}, '': {'type': type(None), 'to_type': lambda x: None, 'to_text': lambda x: ''}}
        xsd_string = 'base64Binary' if base64encode else self.get_type() or type_to_xsd.get(type(value))
        xsd_ns, xsd_type = ['', type(None)] if xsd_string is None else ['', ''] if xsd_string == '' else [XSD if xsd_string in xsd_types_props else '', xsd_string] if ':' not in xsd_string else xsd_string.split(':', 1)
        xsd_type_props = xsd_types_props.get(xsd_type)
        if not xsd_type_props:
            xsd_type_props = xsd_types_props.get('string')
        valid_type = xsd_type_props.get('type', type(None))
        to_type = xsd_type_props.get('to_type', str)
        to_text = xsd_type_props.get('to_text', str)
        if type(value) is str and valid_type is not str:
            try:
                value = to_type(value)
            except (TypeError, ValueError, KeyError):
                _wrong_type_value(xsd=xsd_type, value=value)
        if type(value) is not valid_type:
            _wrong_type_value(xsd=xsd_type, value=value)
        text = to_text(value)
        self.set_type(f'{xsd_ns}:{xsd_type}' if xsd_ns else xsd_type if xsd_type else '')
        SamlBase.__setattr__(self, 'text', text)
        return self

    def harvest_element_tree(self, tree):
        for child in tree:
            self._convert_element_tree_to_member(child)
        for attribute, value in iter(tree.attrib.items()):
            self._convert_element_attribute_to_member(attribute, value)
        text = tree.text.strip() if tree.text and self.extension_elements else tree.text
        if text:
            self.set_text(text)
        if text or self.extension_elements:
            if XSI_NIL in self.extension_attributes:
                del self.extension_attributes[XSI_NIL]