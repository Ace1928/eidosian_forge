import re
import math
import operator
from abc import abstractmethod
from typing import TYPE_CHECKING, cast, Any, List, Optional, Pattern, Union, \
from xml.etree.ElementTree import Element
from elementpath import XPath2Parser, XPathContext, ElementPathError, \
from ..names import XSD_LENGTH, XSD_MIN_LENGTH, XSD_MAX_LENGTH, XSD_ENUMERATION, \
from ..aliases import ElementType, SchemaType, AtomicValueType, BaseXsdType
from ..translation import gettext as _
from ..helpers import count_digits, local_name
from .exceptions import XMLSchemaValidationError, XMLSchemaDecodeError
from .xsdbase import XsdComponent, XsdAnnotation
class XsdExplicitTimezoneFacet(XsdFacet):
    """
    XSD 1.1 *explicitTimezone* facet.

    ..  <explicitTimezone
          fixed = boolean : false
          id = ID
          value = NCName
          {any attributes with non-schema namespace . . .}>
          Content: (annotation?)
        </explicitTimezone>
    """
    value: str
    base_type: BaseXsdType
    _ADMITTED_TAGS = (XSD_EXPLICIT_TIMEZONE,)

    def _parse_value(self, elem: ElementType) -> None:
        self.value = elem.attrib['value']
        if self.value == 'prohibited':
            self._validator = self._prohibited_timezone_validator
        elif self.value == 'required':
            self._validator = self._required_timezone_validator
        elif self.value != 'optional':
            self.value = 'optional'
        facet: Any = self.base_type.get_facet(XSD_EXPLICIT_TIMEZONE)
        if facet is not None and facet.value != self.value and (facet.value != 'optional'):
            msg = _('invalid restriction from {!r}')
            self.parse_error(msg.format(facet.value))

    def _required_timezone_validator(self, value: Any) -> None:
        if value.tzinfo is None:
            reason = _('time zone required for value {!r}').format(self.value)
            raise XMLSchemaValidationError(self, value, reason)

    def _prohibited_timezone_validator(self, value: Any) -> None:
        if value.tzinfo is not None:
            reason = _('time zone prohibited for value {!r}').format(self.value)
            raise XMLSchemaValidationError(self, value, reason)