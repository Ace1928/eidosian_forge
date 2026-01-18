from typing import cast, Any, Callable, Iterator, List, Optional, Tuple, Union
from elementpath.datatypes import AnyAtomicType
from ..exceptions import XMLSchemaValueError
from ..names import XSD_GROUP, XSD_ATTRIBUTE_GROUP, XSD_SEQUENCE, XSD_OVERRIDE, \
from ..aliases import ElementType, NamespacesType, SchemaType, ComponentClassType, \
from ..translation import gettext as _
from ..helpers import get_qname, local_name
from .exceptions import XMLSchemaDecodeError
from .helpers import get_xsd_derivation_attribute
from .xsdbase import XSD_TYPE_DERIVATIONS, XsdComponent, XsdType, ValidationMixin
from .attributes import XsdAttributeGroup
from .assertions import XsdAssert
from .simple_types import FacetsValueType, XsdSimpleType, XsdUnion
from .groups import XsdGroup
from .wildcards import XsdOpenContent, XsdDefaultOpenContent
def _parse_simple_content_extension(self, elem: ElementType, base_type: Any) -> None:
    child = self._parse_child_component(elem, strict=False)
    if child is not None and child.tag not in self._CONTENT_TAIL_TAGS:
        msg = _('unexpected tag %r')
        self.parse_error(msg % child.tag, child)
    if base_type.is_simple():
        self.content = base_type
        self._parse_content_tail(elem)
    else:
        if base_type.has_simple_content():
            self.content = base_type.content
        else:
            self.parse_error(_('base type %r has no simple content') % base_type, elem)
            self.content = self.schema.create_any_content_group(self)
        self._parse_content_tail(elem, derivation='extension', base_attributes=base_type.attributes)