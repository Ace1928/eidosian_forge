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
class XsdEnumerationFacets(MutableSequence[ElementType], XsdFacet):
    """
    Sequence of XSD *enumeration* facets. Values are validates if match any of enumeration values.

    ..  <enumeration
          id = ID
          value = anySimpleType
          {any attributes with non-schema namespace . . .}>
          Content: (annotation?)
        </enumeration>
    """
    base_type: BaseXsdType
    _ADMITTED_TAGS = {XSD_ENUMERATION}

    def __init__(self, elem: ElementType, schema: SchemaType, parent: 'XsdAtomicRestriction', base_type: BaseXsdType) -> None:
        XsdFacet.__init__(self, elem, schema, parent, base_type)

    def _parse(self) -> None:
        self._elements = [self.elem]
        self.enumeration = [self._parse_value(self.elem)]

    def _parse_value(self, elem: ElementType) -> Optional[AtomicValueType]:
        try:
            value = self.base_type.decode(elem.attrib['value'], namespaces=self.schema.namespaces)
        except KeyError:
            pass
        except XMLSchemaValidationError as err:
            self.parse_error(err, elem)
        else:
            if self.base_type.name == XSD_NOTATION_TYPE:
                assert isinstance(value, str)
                try:
                    notation_qname = self.schema.resolve_qname(value)
                except (KeyError, ValueError, RuntimeError) as err:
                    self.parse_error(err, elem)
                else:
                    if notation_qname not in self.maps.notations:
                        msg = _('value {!r} must match a notation declaration')
                        self.parse_error(msg.format(value), elem)
            return cast(AtomicValueType, value)
        return None

    @overload
    @abstractmethod
    def __getitem__(self, i: int) -> ElementType:
        ...

    @overload
    @abstractmethod
    def __getitem__(self, s: slice) -> MutableSequence[ElementType]:
        ...

    def __getitem__(self, i: Union[int, slice]) -> Union[ElementType, MutableSequence[ElementType]]:
        return self._elements[i]

    def __setitem__(self, i: Union[int, slice], o: Any) -> None:
        self._elements[i] = o
        if isinstance(i, int):
            self.enumeration[i] = self._parse_value(o)
        else:
            self.enumeration[i] = [self._parse_value(e) for e in o]

    def __delitem__(self, i: Union[int, slice]) -> None:
        del self._elements[i]
        del self.enumeration[i]

    def __len__(self) -> int:
        return len(self._elements)

    def insert(self, i: int, elem: ElementType) -> None:
        self._elements.insert(i, elem)
        self.enumeration.insert(i, self._parse_value(elem))

    def __repr__(self) -> str:
        if len(self.enumeration) > 5:
            return '%s(%s)' % (self.__class__.__name__, '[%s, ...]' % ', '.join(map(repr, self.enumeration[:5])))
        else:
            return '%s(%r)' % (self.__class__.__name__, self.enumeration)

    def __call__(self, value: Any) -> None:
        if value in self.enumeration:
            return
        try:
            if math.isnan(value):
                if any((math.isnan(x) for x in self.enumeration)):
                    return
            elif math.isinf(value):
                if any((math.isinf(x) and str(value) == str(x) for x in self.enumeration)):
                    return
        except TypeError:
            pass
        reason = _('value must be one of {!r}').format(self.enumeration)
        raise XMLSchemaValidationError(self, value, reason)

    def get_annotation(self, i: int) -> Optional[XsdAnnotation]:
        """
        Get the XSD annotation of the i-th enumeration facet.

        :param i: an integer index.
        :returns: an XsdAnnotation object or `None`.
        """
        for child in self._elements[i]:
            if child.tag == XSD_ANNOTATION:
                return XsdAnnotation(child, self.schema, self)
        return None