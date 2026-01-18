from copy import copy as _copy
from decimal import Decimal
from elementpath.datatypes import AbstractDateTime, Duration, AbstractBinary
from typing import cast, Any, Callable, Union, Dict, List, Optional, \
from ..exceptions import XMLSchemaValueError
from ..names import XSI_NAMESPACE, XSD_ANY_SIMPLE_TYPE, XSD_SIMPLE_TYPE, \
from ..aliases import ComponentClassType, ElementType, IterDecodeType, \
from ..translation import gettext as _
from ..helpers import get_namespace, get_qname
from .exceptions import XMLSchemaValidationError
from .xsdbase import XsdComponent, XsdAnnotation, ValidationMixin
from .simple_types import XsdSimpleType
from .wildcards import XsdAnyAttribute
class XsdAttribute(XsdComponent, ValidationMixin[str, DecodedValueType]):
    """
    Class for XSD 1.0 *attribute* declarations.

    :ivar type: the XSD simpleType of the attribute.

    ..  <attribute
          default = string
          fixed = string
          form = (qualified | unqualified)
          id = ID
          name = NCName
          ref = QName
          type = QName
          use = (optional | prohibited | required) : optional
          {any attributes with non-schema namespace ...}>
          Content: (annotation?, simpleType?)
        </attribute>
    """
    _ADMITTED_TAGS = {XSD_ATTRIBUTE}
    name: str
    local_name: str
    qualified_name: str
    prefixed_name: str
    type: XsdSimpleType
    copy: Callable[['XsdAttribute'], 'XsdAttribute']
    qualified: bool = False
    default: Optional[str] = None
    fixed: Optional[str] = None
    form: Optional[str] = None
    use: str = 'optional'
    inheritable: bool = False

    def _parse(self) -> None:
        attrib = self.elem.attrib
        if 'use' in attrib and self.parent is not None and (attrib['use'] in {'optional', 'prohibited', 'required'}):
            self.use = attrib['use']
        if self._parse_reference():
            try:
                xsd_attribute = self.maps.lookup_attribute(self.name)
            except LookupError:
                self.type = self.any_simple_type
                msg = _('unknown attribute {!r}')
                self.parse_error(msg.format(self.name))
            else:
                self.ref = xsd_attribute
                self.type = xsd_attribute.type
                self.qualified = xsd_attribute.qualified
                self.form = xsd_attribute.form
                if xsd_attribute.default is not None and 'default' not in attrib:
                    self.default = xsd_attribute.default
                if xsd_attribute.fixed is not None:
                    if 'fixed' not in attrib:
                        self.fixed = xsd_attribute.fixed
                    elif xsd_attribute.fixed != attrib['fixed']:
                        msg = _('referenced attribute has a different fixed value {!r}')
                        self.parse_error(msg.format(xsd_attribute.fixed))
            for attribute in ('form', 'type'):
                if attribute in self.elem.attrib:
                    msg = _('attribute {!r} is not allowed when attribute reference is used')
                    self.parse_error(msg.format(attribute))
        else:
            if 'form' in attrib:
                self.form = attrib['form']
                if self.parent is not None and self.form == 'qualified':
                    self.qualified = True
            elif self.schema.attribute_form_default == 'qualified':
                self.qualified = True
            try:
                name = attrib['name']
            except KeyError:
                pass
            else:
                if name == 'xmlns':
                    msg = _("an attribute name must be different from 'xmlns'")
                    self.parse_error(msg)
                if self.parent is None or self.qualified:
                    if self.target_namespace == XSI_NAMESPACE and name not in {'nil', 'type', 'schemaLocation', 'noNamespaceSchemaLocation'}:
                        msg = _('cannot add attributes in %r namespace')
                        self.parse_error(msg % XSI_NAMESPACE)
                    self.name = get_qname(self.target_namespace, name)
                else:
                    self.name = name
            child = self._parse_child_component(self.elem)
            if 'type' in attrib:
                try:
                    type_qname = self.schema.resolve_qname(attrib['type'])
                except (KeyError, ValueError, RuntimeError) as err:
                    self.type = self.any_simple_type
                    self.parse_error(err)
                else:
                    try:
                        self.type = cast(XsdSimpleType, self.maps.lookup_type(type_qname))
                    except LookupError as err:
                        self.type = self.any_simple_type
                        self.parse_error(err)
                    if child is not None and child.tag == XSD_SIMPLE_TYPE:
                        msg = _('ambiguous type definition for XSD attribute')
                        self.parse_error(msg)
            elif child is not None:
                self.type = self.schema.simple_type_factory(child, self.schema, self)
            else:
                self.type = self.any_simple_type
            if not isinstance(self.type, XsdSimpleType):
                self.type = self.any_simple_type
                msg = _("XSD attribute's type must be a simpleType")
                self.parse_error(msg)
        if 'default' in attrib:
            self.default = attrib['default']
            if 'fixed' in attrib:
                msg = _("'default' and 'fixed' attributes are mutually exclusive")
                self.parse_error(msg)
            if self.use != 'optional':
                msg = _("the attribute 'use' must be 'optional' if the attribute 'default' is present")
                self.parse_error(msg)
            if not self.type.is_valid(self.default):
                msg = _("default value {!r} is not compatible with attribute's type")
                self.parse_error(msg.format(self.default))
            elif self.type.is_key() and self.xsd_version == '1.0':
                msg = _('xs:ID key attributes cannot have a default value')
                self.parse_error(msg)
        elif 'fixed' in attrib:
            self.fixed = attrib['fixed']
            if not self.type.is_valid(self.fixed):
                msg = _("fixed value {!r} is not compatible with attribute's type")
                self.parse_error(msg.format(self.fixed))
            elif self.type.is_key() and self.xsd_version == '1.0':
                msg = _('xs:ID key attributes cannot have a fixed value')
                self.parse_error(msg)

    @property
    def built(self) -> bool:
        return True

    @property
    def validation_attempted(self) -> str:
        return 'full'

    @property
    def scope(self) -> str:
        """The scope of the attribute declaration that can be 'global' or 'local'."""
        return 'global' if self.parent is None else 'local'

    @property
    def value_constraint(self) -> Optional[str]:
        """The fixed or the default value if either is defined, `None` otherwise."""
        return self.fixed if self.fixed is not None else self.default

    def is_optional(self) -> bool:
        return self.use == 'optional'

    def is_required(self) -> bool:
        return self.use == 'required'

    def is_prohibited(self) -> bool:
        return self.use == 'prohibited'

    def iter_components(self, xsd_classes: ComponentClassType=None) -> Iterator[XsdComponent]:
        if xsd_classes is None or isinstance(self, xsd_classes):
            yield self
        if self.ref is None and self.type.parent is not None:
            yield from self.type.iter_components(xsd_classes)

    def data_value(self, text: str) -> AtomicValueType:
        """Returns the decoded data value of the provided text as XPath fn:data()."""
        return cast(AtomicValueType, self.decode(text, validation='skip'))

    def iter_decode(self, obj: str, validation: str='lax', **kwargs: Any) -> IterDecodeType[DecodedValueType]:
        if obj is None and self.default is not None:
            obj = self.default
        if self.type.is_notation():
            if self.type.name == XSD_NOTATION_TYPE:
                msg = _('cannot validate against xs:NOTATION directly, only against a subtype with an enumeration facet')
                yield self.validation_error(validation, msg, obj, **kwargs)
            elif not self.type.enumeration:
                msg = _('missing enumeration facet in xs:NOTATION subtype')
                yield self.validation_error(validation, msg, obj, **kwargs)
        if self.fixed is not None:
            if obj is None:
                obj = self.fixed
            elif obj != self.fixed and self.type.text_decode(obj) != self.type.text_decode(self.fixed):
                msg = _('attribute {0!r} has a fixed value {1!r}').format(self.name, self.fixed)
                yield self.validation_error(validation, msg, obj, **kwargs)
        for value in self.type.iter_decode(obj, validation, **kwargs):
            if isinstance(value, XMLSchemaValidationError):
                value.reason = _('attribute {0}={1!r}: {2}').format(self.prefixed_name, obj, value.reason)
                yield value
                continue
            elif 'value_hook' in kwargs:
                yield kwargs['value_hook'](value, self.type)
            elif isinstance(value, (int, float, list)) or value is None:
                yield value
            elif isinstance(value, str):
                if value.startswith('{') and self.type.is_qname():
                    yield obj
                else:
                    yield value
            elif isinstance(value, Decimal):
                try:
                    yield kwargs['decimal_type'](value)
                except (KeyError, TypeError):
                    yield value
            elif isinstance(value, (AbstractDateTime, Duration)):
                yield (value if kwargs.get('datetime_types') else obj.strip())
            elif isinstance(value, AbstractBinary) and (not kwargs.get('binary_types')):
                yield str(value)
            else:
                yield value
            break

    def iter_encode(self, obj: Any, validation: str='lax', **kwargs: Any) -> IterEncodeType[Union[EncodedValueType]]:
        yield from self.type.iter_encode(obj, validation, **kwargs)