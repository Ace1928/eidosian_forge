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
class XsdFacet(XsdComponent):
    """
    XML Schema constraining facets base class.
    """
    value: Optional[AtomicValueType]
    base_type: Optional[BaseXsdType]
    base_value: Optional[AtomicValueType]
    fixed = False

    def __init__(self, elem: ElementType, schema: SchemaType, parent: Union['XsdList', 'XsdAtomicRestriction'], base_type: Optional[BaseXsdType]) -> None:
        self.base_type = base_type
        super(XsdFacet, self).__init__(elem, schema, parent)

    def __repr__(self) -> str:
        return '%s(value=%r, fixed=%r)' % (self.__class__.__name__, self.value, self.fixed)

    def __call__(self, value: Any) -> None:
        try:
            self._validator(value)
        except TypeError:
            reason = _('invalid type {!r} provided').format(type(value))
            raise XMLSchemaValidationError(self, value, reason) from None

    @staticmethod
    def _validator(_: Any) -> None:
        return

    def _parse(self) -> None:
        if 'fixed' in self.elem.attrib and self.elem.attrib['fixed'] in ('true', '1'):
            self.fixed = True
        base_facet = self.base_facet
        self.base_value = None if base_facet is None else base_facet.value
        try:
            self._parse_value(self.elem)
        except (KeyError, ValueError, XMLSchemaDecodeError) as err:
            self.value = None
            self.parse_error(err)
        else:
            if base_facet is not None and base_facet.fixed and (base_facet.value is not None) and (self.value != base_facet.value):
                msg = _('{0!r} facet value is fixed to {1!r}')
                self.parse_error(msg.format(local_name(self.elem.tag), base_facet.value))

    def _parse_value(self, elem: ElementType) -> Union[None, AtomicValueType, Pattern[str]]:
        self.value = elem.attrib['value']
        return None

    @property
    def built(self) -> bool:
        return True

    @property
    def base_facet(self) -> Optional['XsdFacet']:
        """
        An object of the same type if the instance has a base facet, `None` otherwise.
        """
        base_type: Optional[BaseXsdType] = self.base_type
        tag = self.elem.tag
        while True:
            if base_type is None:
                return None
            try:
                base_facet = base_type.facets[tag]
            except (AttributeError, KeyError):
                base_type = base_type.base_type
            else:
                assert isinstance(base_facet, self.__class__)
                return base_facet