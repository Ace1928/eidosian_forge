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
class XsdTotalDigitsFacet(XsdFacet):
    """
    XSD *totalDigits* facet.

    ..  <totalDigits
          fixed = boolean : false
          id = ID
          value = positiveInteger
          {any attributes with non-schema namespace . . .}>
          Content: (annotation?)
        </totalDigits>
    """
    value: int
    base_type: BaseXsdType
    _ADMITTED_TAGS = (XSD_TOTAL_DIGITS,)

    def _parse_value(self, elem: ElementType) -> None:
        try:
            self.value = int(elem.attrib['value'])
        except (ValueError, KeyError):
            self.value = 9999
        else:
            if self.value < 1:
                self.value = 9999
            facet: Any = self.base_type.get_facet(XSD_TOTAL_DIGITS)
            if facet is not None and facet.value < self.value:
                msg = _('invalid restriction: base value is lower ({})')
                self.parse_error(msg.format(facet.value))

    def __call__(self, value: Any) -> None:
        try:
            if operator.add(*count_digits(value)) <= self.value:
                return
        except (TypeError, ValueError, ArithmeticError) as err:
            raise XMLSchemaValidationError(self, value, str(err)) from None
        else:
            reason = _('the number of digits has to be lesser or equal than {!r}').format(self.value)
            raise XMLSchemaValidationError(self, value, reason)