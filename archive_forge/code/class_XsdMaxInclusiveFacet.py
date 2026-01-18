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
class XsdMaxInclusiveFacet(XsdFacet):
    """
    XSD *maxInclusive* facet.

    ..  <maxInclusive
          fixed = boolean : false
          id = ID
          value = anySimpleType
          {any attributes with non-schema namespace . . .}>
          Content: (annotation?)
        </maxInclusive>
    """
    base_type: BaseXsdType
    _ADMITTED_TAGS = (XSD_MAX_INCLUSIVE,)

    def _parse_value(self, elem: ElementType) -> None:
        value = elem.attrib['value']
        self.value, errors = cast(LaxDecodeType, self.base_type.decode(value, 'lax'))
        for e in errors:
            self.parse_error(_('invalid restriction: {}').format(e.reason))

    def __call__(self, value: Any) -> None:
        try:
            if value > self.value:
                reason = _('value has to be less than or equal than {!r}').format(self.value)
                raise XMLSchemaValidationError(self, value, reason)
        except TypeError as err:
            raise XMLSchemaValidationError(self, value, str(err)) from None