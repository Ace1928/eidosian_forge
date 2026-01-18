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
def _parse_content_tail(self, elem: ElementType, **kwargs: Any) -> None:
    self.attributes = self.schema.xsd_attribute_group_class(elem, self.schema, self, **kwargs)
    self.assertions = [XsdAssert(e, self.schema, self, self) for e in elem if e.tag == XSD_ASSERT]
    if isinstance(self.base_type, XsdComplexType):
        self.assertions.extend((XsdAssert(assertion.elem, self.schema, self, self) for assertion in self.base_type.assertions))