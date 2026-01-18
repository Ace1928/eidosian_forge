from decimal import DecimalException
from typing import cast, Any, Callable, Dict, Iterator, List, \
from xml.etree import ElementTree
from ..aliases import ElementType, AtomicValueType, ComponentClassType, \
from ..exceptions import XMLSchemaTypeError, XMLSchemaValueError
from ..names import XSD_NAMESPACE, XSD_ANY_TYPE, XSD_SIMPLE_TYPE, XSD_PATTERN, \
from ..translation import gettext as _
from ..helpers import local_name
from .exceptions import XMLSchemaValidationError, XMLSchemaEncodeError, \
from .xsdbase import XsdComponent, XsdType, ValidationMixin
from .facets import XsdFacet, XsdWhiteSpaceFacet, XsdPatternFacets, \
def _parse_facets(self, facets: Any) -> None:
    base_type: Any
    if facets and self.base_type is not None:
        if isinstance(self.base_type, XsdSimpleType):
            if self.base_type.name == XSD_ANY_SIMPLE_TYPE:
                msg = _('facets not allowed for a direct derivation of xs:anySimpleType')
                self.parse_error(msg)
        elif self.base_type.has_simple_content():
            if self.base_type.content.name == XSD_ANY_SIMPLE_TYPE:
                msg = _('facets not allowed for a direct content derivation of xs:anySimpleType')
                self.parse_error(msg)
    if any((k not in self.admitted_facets for k in facets if k is not None)):
        msg = _('one or more facets are not applicable, admitted set is {!r}')
        self.parse_error(msg.format({local_name(e) for e in self.admitted_facets if e}))
    base_type = {t.base_type for t in facets.values() if isinstance(t, XsdFacet)}
    if len(base_type) > 1:
        msg = _('facet group must have the same base type: %r')
        self.parse_error(msg % base_type)
    base_type = base_type.pop() if base_type else None
    length = getattr(facets.get(XSD_LENGTH), 'value', None)
    min_length = getattr(facets.get(XSD_MIN_LENGTH), 'value', None)
    max_length = getattr(facets.get(XSD_MAX_LENGTH), 'value', None)
    if length is not None:
        if length < 0:
            self.parse_error(_("'length' value must be non a negative integer"))
        if min_length is not None:
            if min_length > length:
                msg = _("'minLength' value must be less than or equal to 'length'")
                self.parse_error(msg)
            min_length_facet = base_type.get_facet(XSD_MIN_LENGTH)
            length_facet = base_type.get_facet(XSD_LENGTH)
            if min_length_facet is None or (length_facet is not None and length_facet.base_type == min_length_facet.base_type):
                msg = _("cannot specify both 'length' and 'minLength'")
                self.parse_error(msg)
        if max_length is not None:
            if max_length < length:
                msg = _("'maxLength' value must be greater or equal to 'length'")
                self.parse_error(msg)
            max_length_facet = base_type.get_facet(XSD_MAX_LENGTH)
            length_facet = base_type.get_facet(XSD_LENGTH)
            if max_length_facet is None or (length_facet is not None and length_facet.base_type == max_length_facet.base_type):
                msg = _("cannot specify both 'length' and 'maxLength'")
                self.parse_error(msg)
        min_length = max_length = length
    elif min_length is not None or max_length is not None:
        min_length_facet = base_type.get_facet(XSD_MIN_LENGTH)
        max_length_facet = base_type.get_facet(XSD_MAX_LENGTH)
        if min_length is not None:
            if min_length < 0:
                msg = _("'minLength' value must be a non negative integer")
                self.parse_error(msg)
            if max_length is not None and max_length < min_length:
                msg = _("'maxLength' value is less than 'minLength'")
                self.parse_error(msg)
            if min_length_facet is not None and min_length_facet.value > min_length:
                msg = _("'minLength' has a lesser value than parent")
                self.parse_error(msg)
            if max_length_facet is not None and min_length > max_length_facet.value:
                msg = _("'minLength' has a greater value than parent 'maxLength'")
                self.parse_error(msg)
        if max_length is not None:
            if max_length < 0:
                msg = _("'maxLength' value must be a non negative integer")
                self.parse_error(msg)
            if min_length_facet is not None and min_length_facet.value > max_length:
                msg = _("'maxLength' has a lesser value than parent 'minLength'")
                self.parse_error(msg)
            if max_length_facet is not None and max_length > max_length_facet.value:
                msg = _("'maxLength' has a greater value than parent")
                self.parse_error(msg)
    min_inclusive = getattr(facets.get(XSD_MIN_INCLUSIVE), 'value', None)
    min_exclusive = getattr(facets.get(XSD_MIN_EXCLUSIVE), 'value', None)
    max_inclusive = getattr(facets.get(XSD_MAX_INCLUSIVE), 'value', None)
    max_exclusive = getattr(facets.get(XSD_MAX_EXCLUSIVE), 'value', None)
    if min_inclusive is not None:
        if min_exclusive is not None:
            msg = _("cannot specify both 'minInclusive' and 'minExclusive'")
            self.parse_error(msg)
        if max_inclusive is not None and min_inclusive > max_inclusive:
            msg = _("'minInclusive' must be less or equal to 'maxInclusive'")
            self.parse_error(msg)
        elif max_exclusive is not None and min_inclusive >= max_exclusive:
            msg = _("'minInclusive' must be lesser than 'maxExclusive'")
            self.parse_error(msg)
    elif min_exclusive is not None:
        if max_inclusive is not None and min_exclusive >= max_inclusive:
            msg = _("'minExclusive' must be lesser than 'maxInclusive'")
            self.parse_error(msg)
        elif max_exclusive is not None and min_exclusive > max_exclusive:
            msg = _("'minExclusive' must be less or equal to 'maxExclusive'")
            self.parse_error(msg)
    if max_inclusive is not None and max_exclusive is not None:
        self.parse_error(_("cannot specify both 'maxInclusive' and 'maxExclusive'"))
    if XSD_TOTAL_DIGITS in facets:
        if XSD_FRACTION_DIGITS in facets and facets[XSD_TOTAL_DIGITS].value < facets[XSD_FRACTION_DIGITS].value:
            msg = _('fractionDigits facet value cannot be lesser than the value of totalDigits facet')
            self.parse_error(msg)
        total_digits = base_type.get_facet(XSD_TOTAL_DIGITS)
        if total_digits is not None and total_digits.value < facets[XSD_TOTAL_DIGITS].value:
            msg = _('totalDigits facet value cannot be greater than the value of the same facet in the base type')
            self.parse_error(msg)
    if XSD_EXPLICIT_TIMEZONE in facets:
        explicit_tz_facet = base_type.get_facet(XSD_EXPLICIT_TIMEZONE)
        if explicit_tz_facet and explicit_tz_facet.value in ('prohibited', 'required') and (facets[XSD_EXPLICIT_TIMEZONE].value != explicit_tz_facet.value):
            msg = _('the explicitTimezone facet value cannot be changed if the base type has the same facet with value %r')
            self.parse_error(msg % explicit_tz_facet.value)
    self.min_length = min_length
    self.max_length = max_length