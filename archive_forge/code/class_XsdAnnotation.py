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
class XsdAnnotation(XsdComponent):
    """
    Class for XSD *annotation* definitions.

    :ivar appinfo: a list containing the xs:appinfo children.
    :ivar documentation: a list containing the xs:documentation children.

    ..  <annotation
          id = ID
          {any attributes with non-schema namespace . . .}>
          Content: (appinfo | documentation)*
        </annotation>

    ..  <appinfo
          source = anyURI
          {any attributes with non-schema namespace . . .}>
          Content: ({any})*
        </appinfo>

    ..  <documentation
          source = anyURI
          xml:lang = language
          {any attributes with non-schema namespace . . .}>
          Content: ({any})*
        </documentation>
    """
    _ADMITTED_TAGS = {XSD_ANNOTATION}
    annotation = None

    def __repr__(self) -> str:
        return '%s(%r)' % (self.__class__.__name__, str(self)[:40])

    def __str__(self) -> str:
        return '\n'.join(select(self.elem, '*/fn:string()'))

    @property
    def built(self) -> bool:
        return True

    def _parse(self) -> None:
        self.appinfo = []
        self.documentation = []
        for child in self.elem:
            if child.tag == XSD_APPINFO:
                self.appinfo.append(child)
            elif child.tag == XSD_DOCUMENTATION:
                self.documentation.append(child)