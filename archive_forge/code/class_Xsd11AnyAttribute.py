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
class Xsd11AnyAttribute(XsdAnyAttribute):
    """
    Class for XSD 1.1 *anyAttribute* declarations.

    ..  <anyAttribute
          id = ID
          namespace = ((##any | ##other) | List of (anyURI | (##targetNamespace | ##local)) )
          notNamespace = List of (anyURI | (##targetNamespace | ##local))
          notQName = List of (QName | ##defined)
          processContents = (lax | skip | strict) : strict
          {any attributes with non-schema namespace . . .}>
          Content: (annotation?)
        </anyAttribute>
    """

    def _parse(self) -> None:
        super(Xsd11AnyAttribute, self)._parse()
        self._parse_not_constraints()

    def is_matching(self, name: Optional[str], default_namespace: Optional[str]=None, **kwargs: Any) -> bool:
        if name is None:
            return False
        elif not name or name[0] == '{':
            namespace = get_namespace(name)
        elif not default_namespace:
            namespace = ''
        else:
            name = f'{{{default_namespace}}}{name}'
            namespace = default_namespace
        if '##defined' in self.not_qname and name in self.maps.attributes:
            xsd_attribute = self.maps.attributes[name]
            if isinstance(xsd_attribute, tuple):
                if xsd_attribute[1] is self.schema:
                    return False
            elif xsd_attribute.schema is self.schema:
                return False
        return name not in self.not_qname and self.is_namespace_allowed(namespace)