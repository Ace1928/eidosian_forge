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
class XsdSimpleType(XsdType, ValidationMixin[Union[str, bytes], DecodedValueType]):
    """
    Base class for simpleTypes definitions. Generally used only for
    instances of xs:anySimpleType.

    ..  <simpleType
          final = (#all | List of (list | union | restriction | extension))
          id = ID
          name = NCName
          {any attributes with non-schema namespace . . .}>
          Content: (annotation?, (restriction | list | union))
        </simpleType>
    """
    _special_types = {XSD_ANY_TYPE, XSD_ANY_SIMPLE_TYPE}
    _ADMITTED_TAGS = {XSD_SIMPLE_TYPE}
    copy: Callable[['XsdSimpleType'], 'XsdSimpleType']
    block: str = ''
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    white_space: Optional[str] = None
    patterns = None
    validators: Union[Tuple[()], List[Union[XsdFacet, Callable[[Any], None]]]] = ()
    allow_empty = True
    facets: Dict[Optional[str], FacetsValueType]
    python_type: PythonTypeClasses
    instance_types: Union[PythonTypeClasses, Tuple[PythonTypeClasses]]
    python_type = instance_types = to_python = from_python = str

    def __init__(self, elem: ElementType, schema: SchemaType, parent: Optional[XsdComponent]=None, name: Optional[str]=None, facets: Optional[Dict[Optional[str], FacetsValueType]]=None) -> None:
        super(XsdSimpleType, self).__init__(elem, schema, parent, name)
        if not hasattr(self, 'facets'):
            self.facets = facets if facets is not None else {}

    def __setattr__(self, name: str, value: Any) -> None:
        super(XsdSimpleType, self).__setattr__(name, value)
        if name == 'facets':
            if not isinstance(self, XsdAtomicBuiltin):
                self._parse_facets(value)
            if self.min_length:
                self.allow_empty = False
            white_space = getattr(self.get_facet(XSD_WHITE_SPACE), 'value', None)
            if white_space is not None:
                self.white_space = white_space
            patterns = self.get_facet(XSD_PATTERN)
            if isinstance(patterns, XsdPatternFacets):
                self.patterns = patterns
                if all((p.match('') is None for p in patterns.patterns)):
                    self.allow_empty = False
            enumeration = self.get_facet(XSD_ENUMERATION)
            if isinstance(enumeration, XsdEnumerationFacets) and '' not in enumeration.enumeration:
                self.allow_empty = False
            if value:
                validators: List[Union[XsdFacet, Callable[[Any], None]]]
                if None in value:
                    validators = [value[None]]
                else:
                    validators = [v for k, v in value.items() if k not in {XSD_WHITE_SPACE, XSD_PATTERN, XSD_ASSERTION}]
                if XSD_ASSERTION in value:
                    assertions: Union[XsdAssertionFacet, List[XsdAssertionFacet]]
                    assertions = value[XSD_ASSERTION]
                    if isinstance(assertions, list):
                        validators.extend(assertions)
                    else:
                        validators.append(assertions)
                if validators:
                    self.validators = validators

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

    @property
    def variety(self) -> Optional[str]:
        return None

    @property
    def simple_type(self) -> 'XsdSimpleType':
        return self

    @property
    def min_value(self) -> Optional[AtomicValueType]:
        min_exclusive: Optional['AtomicValueType']
        min_inclusive: Optional['AtomicValueType']
        min_exclusive = cast(Optional['AtomicValueType'], getattr(self.get_facet(XSD_MIN_EXCLUSIVE), 'value', None))
        min_inclusive = cast(Optional['AtomicValueType'], getattr(self.get_facet(XSD_MIN_INCLUSIVE), 'value', None))
        if min_exclusive is None:
            return min_inclusive
        elif min_inclusive is None:
            return min_exclusive
        elif min_inclusive <= min_exclusive:
            return min_exclusive
        else:
            return min_inclusive

    @property
    def max_value(self) -> Optional[AtomicValueType]:
        max_exclusive: Optional['AtomicValueType']
        max_inclusive: Optional['AtomicValueType']
        max_exclusive = cast(Optional['AtomicValueType'], getattr(self.get_facet(XSD_MAX_EXCLUSIVE), 'value', None))
        max_inclusive = cast(Optional['AtomicValueType'], getattr(self.get_facet(XSD_MAX_INCLUSIVE), 'value', None))
        if max_exclusive is None:
            return max_inclusive
        elif max_inclusive is None:
            return max_exclusive
        elif max_inclusive >= max_exclusive:
            return max_exclusive
        else:
            return max_inclusive

    @property
    def enumeration(self) -> Optional[List[Optional[AtomicValueType]]]:
        enumeration = self.get_facet(XSD_ENUMERATION)
        if isinstance(enumeration, XsdEnumerationFacets):
            return enumeration.enumeration
        return None

    @property
    def admitted_facets(self) -> Set[str]:
        return XSD_10_FACETS if self.xsd_version == '1.0' else XSD_11_FACETS

    @property
    def built(self) -> bool:
        return True

    @staticmethod
    def is_simple() -> bool:
        return True

    @staticmethod
    def is_complex() -> bool:
        return False

    @property
    def content_type_label(self) -> str:
        return 'empty' if self.max_length == 0 else 'simple'

    @property
    def sequence_type(self) -> str:
        if self.is_empty():
            return 'empty-sequence()'
        root_type = self.root_type
        if root_type.name is not None:
            sequence_type = f'xs:{root_type.local_name}'
        else:
            sequence_type = 'xs:untypedAtomic'
        if not self.is_list():
            return sequence_type
        elif self.is_emptiable():
            return f'{sequence_type}*'
        else:
            return f'{sequence_type}+'

    def is_empty(self) -> bool:
        return self.max_length == 0 or (self.enumeration is not None and all((v == '' for v in self.enumeration)))

    def is_emptiable(self) -> bool:
        return self.allow_empty

    def has_simple_content(self) -> bool:
        return self.max_length != 0

    def has_complex_content(self) -> bool:
        return False

    def has_mixed_content(self) -> bool:
        return False

    def is_element_only(self) -> bool:
        return False

    def is_derived(self, other: Union[BaseXsdType, Tuple[ElementType, SchemaType]], derivation: Optional[str]=None) -> bool:
        if self is other:
            return True
        elif isinstance(other, tuple):
            other[1].parse_error(f'global type {other[0].tag!r} is not built')
            return False
        elif derivation and self.derivation and (derivation != self.derivation):
            return False
        elif other.name in self._special_types:
            return derivation != 'extension'
        elif self.base_type is other:
            return True
        elif self.base_type is None:
            if isinstance(other, XsdUnion):
                return any((self.is_derived(m, derivation) for m in other.member_types))
            return False
        elif self.base_type.is_complex():
            if not self.base_type.has_simple_content():
                return False
            return self.base_type.content.is_derived(other, derivation)
        elif isinstance(other, XsdUnion):
            return any((self.is_derived(m, derivation) for m in other.member_types))
        else:
            return self.base_type.is_derived(other, derivation)

    def is_dynamic_consistent(self, other: BaseXsdType) -> bool:
        return other.name in {XSD_ANY_TYPE, XSD_ANY_SIMPLE_TYPE} or self.is_derived(other) or (isinstance(other, XsdUnion) and any((self.is_derived(mt) for mt in other.member_types)))

    def normalize(self, text: Union[str, bytes]) -> str:
        """
        Normalize and restrict value-space with pre-lexical and lexical facets.

        :param text: text string encoded value.
        :return: a normalized string.
        """
        if isinstance(text, bytes):
            text = text.decode('utf-8')
        elif not isinstance(text, str):
            raise XMLSchemaValueError('argument is not a string: %r' % text)
        if self.white_space == 'replace':
            return self._REGEX_SPACE.sub(' ', text)
        elif self.white_space == 'collapse':
            return self._REGEX_SPACES.sub(' ', text).strip()
        else:
            return text

    def text_decode(self, text: str) -> AtomicValueType:
        return cast(AtomicValueType, self.decode(text, validation='skip'))

    def iter_decode(self, obj: Union[str, bytes], validation: str='lax', **kwargs: Any) -> IterDecodeType[DecodedValueType]:
        text = self.normalize(obj)
        if self.patterns is not None:
            try:
                self.patterns(text)
            except XMLSchemaValidationError as err:
                yield err
        for validator in self.validators:
            try:
                validator(text)
            except XMLSchemaValidationError as err:
                yield err
        yield text

    def iter_encode(self, obj: Any, validation: str='lax', **kwargs: Any) -> IterEncodeType[EncodedValueType]:
        if isinstance(obj, (str, bytes)):
            text = self.normalize(obj)
        elif obj is None:
            text = ''
        elif isinstance(obj, list):
            text = ' '.join((str(x) for x in obj))
        else:
            text = str(obj)
        if self.patterns is not None:
            try:
                self.patterns(text)
            except XMLSchemaValidationError as err:
                yield err
        for validator in self.validators:
            try:
                validator(text)
            except XMLSchemaValidationError as err:
                yield err
        yield text

    def get_facet(self, tag: str) -> Optional[FacetsValueType]:
        return self.facets.get(tag)