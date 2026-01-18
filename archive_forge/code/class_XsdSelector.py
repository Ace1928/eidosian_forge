import re
import math
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Pattern, \
from elementpath import XPath2Parser, ElementPathError, XPathToken, XPathContext, \
from ..exceptions import XMLSchemaTypeError, XMLSchemaValueError
from ..names import XSD_QNAME, XSD_UNIQUE, XSD_KEY, XSD_KEYREF, XSD_SELECTOR, XSD_FIELD
from ..translation import gettext as _
from ..helpers import get_qname, get_extended_qname
from ..aliases import ElementType, SchemaType, NamespacesType, AtomicValueType
from .exceptions import XMLSchemaNotBuiltError
from .xsdbase import XsdComponent
from .attributes import XsdAttribute
from .wildcards import XsdAnyElement
from . import elements
class XsdSelector(XsdComponent):
    """Class for defining an XPath selector for an XSD identity constraint."""
    _ADMITTED_TAGS = {XSD_SELECTOR}
    xpath_default_namespace = ''
    pattern: Union[str, Pattern[str]] = translate_pattern('(\\.//)?(((child::)?((\\i\\c*:)?(\\i\\c*|\\*)))|\\.)(/(((child::)?((\\i\\c*:)?(\\i\\c*|\\*)))|\\.))*(\\|(\\.//)?(((child::)?((\\i\\c*:)?(\\i\\c*|\\*)))|\\.)(/(((child::)?((\\i\\c*:)?(\\i\\c*|\\*)))|\\.))*)*', back_references=False, lazy_quantifiers=False, anchors=False)
    token: Optional[XPathToken] = None
    parser: Optional[IdentityXPathParser] = None

    def __init__(self, elem: ElementType, schema: SchemaType, parent: Optional['XsdIdentity']) -> None:
        super(XsdSelector, self).__init__(elem, schema, parent)

    def _parse(self) -> None:
        try:
            self.path = self.elem.attrib['xpath']
        except KeyError:
            self.parse_error(_("'xpath' attribute required"))
            self.path = '*'
        else:
            path = self.path.replace(' ', '')
            try:
                _match = self.pattern.match(path)
            except AttributeError:
                self.__class__.pattern = re.compile(self.pattern)
                _match = self.pattern.match(path)
            if not _match:
                msg = _('invalid XPath expression for an {}')
                self.parse_error(msg.format(self.__class__.__name__))
        if self.schema.XSD_VERSION > '1.0':
            if 'xpathDefaultNamespace' in self.elem.attrib:
                self.xpath_default_namespace = self._parse_xpath_default_namespace(self.elem)
            else:
                self.xpath_default_namespace = self.schema.xpath_default_namespace
        self.parser = IdentityXPathParser(namespaces=self.namespaces, strict=False, compatibility_mode=True, default_namespace=self.xpath_default_namespace)
        try:
            self.token = self.parser.parse(self.path)
        except ElementPathError as err:
            self.token = self.parser.parse('*')
            self.parse_error(err)

    def __repr__(self) -> str:
        return '%s(path=%r)' % (self.__class__.__name__, self.path)

    @property
    def built(self) -> bool:
        return self.token is not None

    @property
    def target_namespace(self) -> str:
        if self.token is None:
            pass
        elif self.token.symbol == ':':
            return self.token[1].namespace or self.xpath_default_namespace
        elif self.token.symbol == '@' and self.token[0].symbol == ':':
            return self.token[0][1].namespace or self.xpath_default_namespace
        return self.schema.target_namespace