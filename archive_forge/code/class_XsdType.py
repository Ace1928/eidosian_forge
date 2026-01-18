import re
from typing import TYPE_CHECKING, cast, Any, Dict, Generic, List, Iterator, Optional, \
from xml.etree import ElementTree
from elementpath import select
from elementpath.etree import is_etree_element, etree_tostring
from ..exceptions import XMLSchemaValueError, XMLSchemaTypeError
from ..names import XSD_ANNOTATION, XSD_APPINFO, XSD_DOCUMENTATION, \
from ..aliases import ElementType, NamespacesType, SchemaType, BaseXsdType, \
from ..translation import gettext as _
from ..helpers import get_qname, local_name, get_prefixed_qname
from ..resources import XMLResource
from .exceptions import XMLSchemaParseError, XMLSchemaValidationError
class XsdType(XsdComponent):
    """Common base class for XSD types."""
    abstract = False
    base_type: Optional[BaseXsdType] = None
    derivation: Optional[str] = None
    _final: Optional[str] = None

    @property
    def final(self) -> str:
        return self.schema.final_default if self._final is None else self._final

    @property
    def built(self) -> bool:
        raise NotImplementedError()

    @property
    def content_type_label(self) -> str:
        """The content type classification."""
        raise NotImplementedError()

    @property
    def sequence_type(self) -> str:
        """The XPath sequence type associated with the content."""
        raise NotImplementedError()

    @property
    def root_type(self) -> BaseXsdType:
        """
        The root type of the type definition hierarchy. For an atomic type
        is the primitive type. For a list is the primitive type of the item.
        For a union is the base union type. For a complex type is xs:anyType.
        """
        if getattr(self, 'attributes', None):
            return cast('XsdComplexType', self.maps.types[XSD_ANY_TYPE])
        elif self.base_type is None:
            if self.is_simple():
                return cast('XsdSimpleType', self)
            return cast('XsdComplexType', self.maps.types[XSD_ANY_TYPE])
        primitive_type: BaseXsdType
        try:
            if self.base_type.is_simple():
                primitive_type = self.base_type.primitive_type
            else:
                primitive_type = self.base_type.content.primitive_type
        except AttributeError:
            return self.base_type.root_type
        else:
            return primitive_type

    @property
    def simple_type(self) -> Optional['XsdSimpleType']:
        """
        Property that is the instance itself for a simpleType. For a
        complexType is the instance's *content* if this is a simpleType
        or `None` if the instance's *content* is a model group.
        """
        raise NotImplementedError()

    @property
    def model_group(self) -> Optional['XsdGroup']:
        """
        Property that is `None` for a simpleType. For a complexType is
        the instance's *content* if this is a model group or `None` if
        the instance's *content* is a simpleType.
        """
        return None

    @staticmethod
    def is_simple() -> bool:
        """Returns `True` if the instance is a simpleType, `False` otherwise."""
        raise NotImplementedError()

    @staticmethod
    def is_complex() -> bool:
        """Returns `True` if the instance is a complexType, `False` otherwise."""
        raise NotImplementedError()

    def is_atomic(self) -> bool:
        """Returns `True` if the instance is an atomic simpleType, `False` otherwise."""
        return False

    def is_list(self) -> bool:
        """Returns `True` if the instance is a list simpleType, `False` otherwise."""
        return False

    def is_union(self) -> bool:
        """Returns `True` if the instance is a union simpleType, `False` otherwise."""
        return False

    def is_datetime(self) -> bool:
        """
        Returns `True` if the instance is a datetime/duration XSD builtin-type, `False` otherwise.
        """
        return False

    def is_empty(self) -> bool:
        """Returns `True` if the instance has an empty content, `False` otherwise."""
        raise NotImplementedError()

    def is_emptiable(self) -> bool:
        """Returns `True` if the instance has an emptiable value or content, `False` otherwise."""
        raise NotImplementedError()

    def has_simple_content(self) -> bool:
        """
        Returns `True` if the instance has a simple content, `False` otherwise.
        """
        raise NotImplementedError()

    def has_complex_content(self) -> bool:
        """
        Returns `True` if the instance is a complexType with mixed or element-only
        content, `False` otherwise.
        """
        raise NotImplementedError()

    def has_mixed_content(self) -> bool:
        """
        Returns `True` if the instance is a complexType with mixed content, `False` otherwise.
        """
        raise NotImplementedError()

    def is_element_only(self) -> bool:
        """
        Returns `True` if the instance is a complexType with element-only content,
        `False` otherwise.
        """
        raise NotImplementedError()

    def is_derived(self, other: Union[BaseXsdType, Tuple[ElementType, SchemaType]], derivation: Optional[str]=None) -> bool:
        """
        Returns `True` if the instance is derived from *other*, `False` otherwise.
        The optional argument derivation can be a string containing the words
        'extension' or 'restriction' or both.
        """
        raise NotImplementedError()

    def is_extension(self) -> bool:
        return self.derivation == 'extension'

    def is_restriction(self) -> bool:
        return self.derivation == 'restriction'

    def is_blocked(self, xsd_element: 'XsdElement') -> bool:
        """
        Returns `True` if the base type derivation is blocked, `False` otherwise.
        """
        xsd_type = xsd_element.type
        if self is xsd_type:
            return False
        block = f'{xsd_element.block} {xsd_type.block}'.strip()
        if not block:
            return False
        _block = {x for x in block.split() if x in ('extension', 'restriction')}
        return any((self.is_derived(xsd_type, derivation) for derivation in _block))

    def is_dynamic_consistent(self, other: Any) -> bool:
        return other.name == XSD_ANY_TYPE or self.is_derived(other) or (hasattr(other, 'member_types') and any((self.is_derived(mt) for mt in other.member_types)))

    def is_key(self) -> bool:
        return self.name == XSD_ID or self.is_derived(self.maps.types[XSD_ID])

    def is_qname(self) -> bool:
        return self.name == XSD_QNAME or self.is_derived(self.maps.types[XSD_QNAME])

    def is_notation(self) -> bool:
        return self.name == XSD_NOTATION_TYPE or self.is_derived(self.maps.types[XSD_NOTATION_TYPE])

    def is_decimal(self) -> bool:
        return self.name == XSD_DECIMAL or self.is_derived(self.maps.types[XSD_DECIMAL])

    def text_decode(self, text: str) -> Any:
        raise NotImplementedError()