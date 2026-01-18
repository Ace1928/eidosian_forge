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
class XsdPatternFacets(MutableSequence[ElementType], XsdFacet):
    """
    Sequence of XSD *pattern* facets. Values are validates if match any of patterns.

    ..  <pattern
          id = ID
          value = string
          {any attributes with non-schema namespace . . .}>
          Content: (annotation?)
        </pattern>
    """
    _ADMITTED_TAGS = {XSD_PATTERN}
    patterns: List[Pattern[str]]

    def __init__(self, elem: ElementType, schema: SchemaType, parent: 'XsdAtomicRestriction', base_type: Optional[BaseXsdType]) -> None:
        XsdFacet.__init__(self, elem, schema, parent, base_type)

    def _parse(self) -> None:
        self._elements = [self.elem]
        self.patterns = [self._parse_value(self.elem)]

    def _parse_value(self, elem: ElementType) -> Pattern[str]:
        try:
            python_pattern = translate_pattern(pattern=elem.attrib['value'], xsd_version=self.xsd_version, back_references=False, lazy_quantifiers=False, anchors=False)
            return re.compile(python_pattern)
        except KeyError:
            return re.compile('^.*$')
        except (RegexError, re.error, XMLSchemaDecodeError) as err:
            self.parse_error(str(err), elem)
            return re.compile('^.*$')

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
            self.patterns[i] = self._parse_value(o)
        else:
            self.patterns[i] = [self._parse_value(e) for e in o]

    def __delitem__(self, i: Union[int, slice]) -> None:
        del self._elements[i]
        del self.patterns[i]

    def __len__(self) -> int:
        return len(self._elements)

    def insert(self, i: int, elem: ElementType) -> None:
        self._elements.insert(i, elem)
        self.patterns.insert(i, self._parse_value(elem))

    def __repr__(self) -> str:
        s = repr(self.regexps)
        if len(s) < 70:
            return '%s(%s)' % (self.__class__.__name__, s)
        else:
            return "%s(%s...'])" % (self.__class__.__name__, s[:70])

    def __call__(self, text: str) -> None:
        try:
            if all((pattern.match(text) is None for pattern in self.patterns)):
                reason = _("value doesn't match any pattern of {!r}").format(self.regexps)
                raise XMLSchemaValidationError(self, text, reason)
        except TypeError as err:
            raise XMLSchemaValidationError(self, text, str(err)) from None

    @property
    def regexps(self) -> List[str]:
        return [e.attrib.get('value', '') for e in self._elements]

    def get_annotation(self, i: int) -> Optional[XsdAnnotation]:
        """
        Get the XSD annotation of the i-th pattern facet.

        :param i: an integer index.
        :returns: an XsdAnnotation object or `None`.
        """
        for child in self._elements[i]:
            if child.tag == XSD_ANNOTATION:
                return XsdAnnotation(child, self.schema, self)
        return None