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
def _parse_simple_content_restriction(self, elem: ElementType, base_type: Any) -> None:
    if base_type.is_simple():
        msg = _('a complexType ancestor required: {!r}')
        self.parse_error(msg.format(base_type), elem)
        self.content = self.schema.create_any_content_group(self)
        self._parse_content_tail(elem)
    else:
        if base_type.is_empty():
            self.content = self.schema.xsd_atomic_restriction_class(elem, self.schema, self)
            if not self.is_empty():
                msg = _('a not empty simpleContent cannot restrict an empty content type')
                self.parse_error(msg, elem)
                self.content = self.schema.create_any_content_group(self)
        elif base_type.has_simple_content():
            self.content = self.schema.xsd_atomic_restriction_class(elem, self.schema, self)
            if not self.content.is_derived(base_type.content, 'restriction'):
                msg = _('content type is not a restriction of base content')
                self.parse_error(msg, elem)
        elif base_type.mixed and base_type.is_emptiable():
            self.content = self.schema.xsd_atomic_restriction_class(elem, self.schema, self)
        else:
            msg = _('with simpleContent cannot restrict an element-only content type')
            self.parse_error(msg, elem)
            self.content = self.schema.create_any_content_group(self)
        self._parse_content_tail(elem, derivation='restriction', base_attributes=base_type.attributes)