from typing import cast, Any, Callable, Dict, Iterable, Iterator, List, Optional, \
from elementpath import SchemaElementNode, build_schema_node_tree
from ..exceptions import XMLSchemaValueError
from ..names import XSI_NAMESPACE, XSD_ANY, XSD_ANY_ATTRIBUTE, \
from ..aliases import ElementType, SchemaType, SchemaElementType, SchemaAttributeType, \
from ..translation import gettext as _
from ..helpers import get_namespace, raw_xml_encode
from ..xpath import XsdSchemaProtocol, XsdElementProtocol, XMLSchemaProxy, ElementPathMixin
from .xsdbase import ValidationMixin, XsdComponent
from .particles import ParticleMixin
from . import elements
class XsdDefaultOpenContent(XsdOpenContent):
    """
    Class for XSD 1.1 *defaultOpenContent* model definitions.

    ..  <defaultOpenContent
          appliesToEmpty = boolean : false
          id = ID
          mode = (interleave | suffix) : interleave
          {any attributes with non-schema namespace . . .}>
          Content: (annotation?, any)
        </defaultOpenContent>
    """
    _ADMITTED_TAGS = {XSD_DEFAULT_OPEN_CONTENT}
    applies_to_empty = False

    def __init__(self, elem: ElementType, schema: SchemaType) -> None:
        super(XsdOpenContent, self).__init__(elem, schema)

    def _parse(self) -> None:
        super(XsdDefaultOpenContent, self)._parse()
        if self.parent is not None:
            msg = _('defaultOpenContent must be a child of the schema')
            self.parse_error(msg)
        if self.mode == 'none':
            msg = _("the attribute 'mode' of a defaultOpenContent cannot be 'none'")
            self.parse_error(msg)
        if self._parse_child_component(self.elem) is None:
            msg = _('a defaultOpenContent declaration cannot be empty')
            self.parse_error(msg)
        if 'appliesToEmpty' in self.elem.attrib:
            if self.elem.attrib['appliesToEmpty'].strip() in {'true', '1'}:
                self.applies_to_empty = True