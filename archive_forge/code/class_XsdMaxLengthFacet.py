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
class XsdMaxLengthFacet(XsdFacet):
    """
    XSD *maxLength* facet.

    ..  <maxLength
          fixed = boolean : false
          id = ID
          value = nonNegativeInteger
          {any attributes with non-schema namespace . . .}>
          Content: (annotation?)
        </maxLength>
    """
    value: int
    base_type: BaseXsdType
    base_value: Optional[int]
    _ADMITTED_TAGS = (XSD_MAX_LENGTH,)

    def _parse_value(self, elem: ElementType) -> None:
        self.value = int(elem.attrib['value'])
        if self.base_value is not None and self.value > self.base_value:
            msg = _('base type has a lesser max length ({})')
            self.parse_error(msg.format(self.base_value))
        primitive_type = getattr(self.base_type, 'primitive_type', None)
        if primitive_type is None or primitive_type.name not in {XSD_QNAME, XSD_NOTATION_TYPE}:
            self._validator = self._max_length_validator

    def _max_length_validator(self, value: Any) -> None:
        if len(value) > self.value:
            reason = _('value length cannot be greater than {!r}').format(self.value)
            raise XMLSchemaValidationError(self, value, reason)