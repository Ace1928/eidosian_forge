import warnings
from copy import copy as _copy
from decimal import Decimal
from types import GeneratorType
from typing import TYPE_CHECKING, cast, Any, Dict, Iterator, List, Optional, \
from xml.etree.ElementTree import Element
from elementpath import XPath2Parser, ElementPathError, XPathContext, XPathToken, \
from elementpath.datatypes import AbstractDateTime, Duration, AbstractBinary
from ..exceptions import XMLSchemaTypeError, XMLSchemaValueError
from ..names import XSD_COMPLEX_TYPE, XSD_SIMPLE_TYPE, XSD_ALTERNATIVE, \
from ..aliases import ElementType, SchemaType, BaseXsdType, SchemaElementType, \
from ..translation import gettext as _
from ..helpers import get_qname, get_namespace, etree_iter_location_hints, \
from .. import dataobjects
from ..converters import ElementData, XMLSchemaConverter
from ..xpath import XsdSchemaProtocol, XsdElementProtocol, XMLSchemaProxy, \
from ..resources import XMLResource
from .exceptions import XMLSchemaNotBuiltError, XMLSchemaValidationError, \
from .helpers import get_xsd_derivation_attribute
from .xsdbase import XSD_TYPE_DERIVATIONS, XSD_ELEMENT_DERIVATIONS, \
from .particles import ParticleMixin, OccursCalculator
from .identities import XsdIdentity, XsdKey, XsdUnique, \
from .simple_types import XsdSimpleType
from .attributes import XsdAttribute
from .wildcards import XsdAnyElement
class XsdAlternative(XsdComponent):
    """
    XSD 1.1 type *alternative* definitions.

    ..  <alternative
          id = ID
          test = an XPath expression
          type = QName
          xpathDefaultNamespace = (anyURI | (##defaultNamespace | ##targetNamespace | ##local))
          {any attributes with non-schema namespace . . .}>
          Content: (annotation?, (simpleType | complexType)?)
        </alternative>
    """
    parent: XsdElement
    type: BaseXsdType
    path: Optional[str] = None
    token: Optional[XPathToken] = None
    _ADMITTED_TAGS = {XSD_ALTERNATIVE}

    def __init__(self, elem: ElementType, schema: SchemaType, parent: XsdElement) -> None:
        super(XsdAlternative, self).__init__(elem, schema, parent)

    def __repr__(self) -> str:
        return '%s(type=%r, test=%r)' % (self.__class__.__name__, self.elem.get('type'), self.elem.get('test'))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, XsdAlternative) and self.path == other.path and (self.type is other.type) and (self.xpath_default_namespace == other.xpath_default_namespace)

    def __ne__(self, other: object) -> bool:
        return not isinstance(other, XsdAlternative) or self.path != other.path or self.type is not other.type or (self.xpath_default_namespace != other.xpath_default_namespace)

    def _parse(self) -> None:
        attrib = self.elem.attrib
        if 'xpathDefaultNamespace' in attrib:
            self.xpath_default_namespace = self._parse_xpath_default_namespace(self.elem)
        else:
            self.xpath_default_namespace = self.schema.xpath_default_namespace
        parser = XPath2Parser(namespaces=self.namespaces, strict=False, default_namespace=self.xpath_default_namespace)
        try:
            self.path = attrib['test']
        except KeyError:
            pass
        else:
            try:
                self.token = parser.parse(self.path)
            except ElementPathError as err:
                self.parse_error(err)
                self.token = parser.parse('false()')
                self.path = 'false()'
        try:
            type_qname = self.schema.resolve_qname(attrib['type'])
        except (KeyError, ValueError, RuntimeError) as err:
            if 'type' in attrib:
                self.parse_error(err)
                self.type = self.any_type
            else:
                child = self._parse_child_component(self.elem, strict=False)
                if child is None or child.tag not in (XSD_COMPLEX_TYPE, XSD_SIMPLE_TYPE):
                    self.parse_error(_("missing 'type' attribute"))
                    self.type = self.any_type
                elif child.tag == XSD_COMPLEX_TYPE:
                    self.type = self.schema.xsd_complex_type_class(child, self.schema, self)
                else:
                    self.type = self.schema.simple_type_factory(child, self.schema, self)
                if not self.type.is_derived(self.parent.type):
                    msg = _('declared type is not derived from {!r}')
                    self.parse_error(msg.format(self.parent.type))
        else:
            try:
                self.type = self.maps.lookup_type(type_qname)
            except KeyError:
                self.parse_error(_('unknown type {!r}').format(attrib['type']))
                self.type = self.any_type
            else:
                if self.type.name != XSD_ERROR and (not self.type.is_derived(self.parent.type)):
                    msg = _('type {0!r} is not derived from {1!r}')
                    self.parse_error(msg.format(attrib['type'], self.parent.type))
                child = self._parse_child_component(self.elem, strict=False)
                if child is not None and child.tag in (XSD_COMPLEX_TYPE, XSD_SIMPLE_TYPE):
                    msg = _("the attribute 'type' and the xs:%s local declaration are mutually exclusive")
                    self.parse_error(msg % child.tag.split('}')[-1])

    @property
    def built(self) -> bool:
        if not hasattr(self, 'type'):
            return False
        return self.type.parent is None or self.type.built

    @property
    def validation_attempted(self) -> str:
        if self.built:
            return 'full'
        elif not hasattr(self, 'type'):
            return 'none'
        else:
            return self.type.validation_attempted

    def iter_components(self, xsd_classes: ComponentClassType=None) -> Iterator[XsdComponent]:
        if xsd_classes is None or isinstance(self, xsd_classes):
            yield self
        if self.type is not None and self.type.parent is not None:
            yield from self.type.iter_components(xsd_classes)

    def test(self, elem: ElementType) -> bool:
        if self.token is None:
            return False
        try:
            result = list(self.token.select(context=XPathContext(elem)))
            return self.token.boolean_value(result)
        except (TypeError, ValueError):
            return False