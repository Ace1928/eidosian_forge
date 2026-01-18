import warnings
from collections import Counter
from functools import lru_cache
from typing import cast, Any, Callable, Dict, List, Iterable, Iterator, \
from ..exceptions import XMLSchemaKeyError, XMLSchemaTypeError, \
from ..names import XSD_NAMESPACE, XSD_REDEFINE, XSD_OVERRIDE, XSD_NOTATION, \
from ..aliases import ComponentClassType, ElementType, SchemaType, BaseXsdType, \
from ..helpers import get_qname, local_name, get_extended_qname
from ..namespaces import NamespaceResourcesMap
from ..translation import gettext as _
from .exceptions import XMLSchemaNotBuiltError, XMLSchemaModelError, XMLSchemaModelDepthError, \
from .xsdbase import XsdValidator, XsdComponent
from .builtins import xsd_builtin_types_factory
from .models import check_model
from . import XsdAttribute, XsdSimpleType, XsdComplexType, XsdElement, XsdAttributeGroup, \
def _build_global(self, obj: Any, qname: str, global_map: Dict[str, Any]) -> Any:
    factory_or_class: Callable[[ElementType, SchemaType], Any]
    if isinstance(obj, tuple):
        try:
            elem, schema = obj
        except ValueError:
            return obj[0]
        try:
            factory_or_class = self._builders[elem.tag]
        except KeyError:
            msg = _('wrong element {0!r} for map {1!r}')
            raise XMLSchemaKeyError(msg.format(elem, global_map))
        global_map[qname] = (obj,)
        global_map[qname] = factory_or_class(elem, schema)
        return global_map[qname]
    elif isinstance(obj, list):
        try:
            elem, schema = obj[0]
        except ValueError:
            return obj[0][0]
        try:
            factory_or_class = self._builders[elem.tag]
        except KeyError:
            msg = _('wrong element {0!r} for map {1!r}')
            raise XMLSchemaKeyError(msg.format(elem, global_map))
        global_map[qname] = (obj[0],)
        global_map[qname] = component = factory_or_class(elem, schema)
        for elem, schema in obj[1:]:
            if component.schema.target_namespace != schema.target_namespace:
                msg = _('redefined schema {!r} has a different targetNamespace')
                raise XMLSchemaValueError(msg.format(schema))
            component.redefine = component.copy()
            component.redefine.parent = component
            component.schema = schema
            component.elem = elem
        return global_map[qname]
    else:
        msg = _('unexpected instance {!r} in global map')
        raise XMLSchemaTypeError(msg.format(obj))