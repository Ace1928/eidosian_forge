from abc import ABCMeta
import os
import logging
import threading
import warnings
import re
import sys
from copy import copy as _copy
from operator import attrgetter
from typing import cast, Callable, ItemsView, List, Optional, Dict, Any, \
from xml.etree.ElementTree import Element, ParseError
from elementpath import XPathToken, SchemaElementNode, build_schema_node_tree
from ..exceptions import XMLSchemaTypeError, XMLSchemaKeyError, XMLSchemaRuntimeError, \
from ..names import VC_MIN_VERSION, VC_MAX_VERSION, VC_TYPE_AVAILABLE, \
from ..aliases import ElementType, XMLSourceType, NamespacesType, LocationsType, \
from ..translation import gettext as _
from ..helpers import prune_etree, get_namespace, get_qname, is_defuse_error
from ..namespaces import NamespaceResourcesMap, NamespaceView
from ..resources import is_local_url, is_remote_url, url_path_is_file, \
from ..converters import XMLSchemaConverter
from ..xpath import XsdSchemaProtocol, XMLSchemaProxy, ElementPathMixin
from .. import dataobjects
from .exceptions import XMLSchemaParseError, XMLSchemaValidationError, XMLSchemaEncodeError, \
from .helpers import get_xsd_derivation_attribute
from .xsdbase import check_validation_mode, XsdValidator, XsdComponent, XsdAnnotation
from .notations import XsdNotation
from .identities import XsdIdentity, XsdKey, XsdKeyref, XsdUnique, \
from .facets import XSD_10_FACETS, XSD_11_FACETS
from .simple_types import XsdSimpleType, XsdList, XsdUnion, XsdAtomicRestriction, \
from .attributes import XsdAttribute, XsdAttributeGroup, Xsd11Attribute
from .complex_types import XsdComplexType, Xsd11ComplexType
from .groups import XsdGroup, Xsd11Group
from .elements import XsdElement, Xsd11Element
from .wildcards import XsdAnyElement, XsdAnyAttribute, Xsd11AnyElement, \
from .global_maps import XsdGlobals
class XMLSchemaMeta(ABCMeta):
    XSD_VERSION: str
    create_meta_schema: Callable[['XMLSchemaBase', Optional[str]], SchemaType]

    def __new__(mcs, name: str, bases: Tuple[Type[Any], ...], dict_: Dict[str, Any]) -> 'XMLSchemaMeta':
        assert bases, 'a base class is mandatory'
        base_class = bases[0]
        if isinstance(dict_.get('meta_schema'), str):
            meta_schema_file: str = dict_.pop('meta_schema')
            meta_schema_class_name = 'Meta' + name
            meta_schema: Optional[SchemaType]
            meta_schema = getattr(base_class, 'meta_schema', None)
            if meta_schema is None:
                meta_bases = bases
            else:
                meta_bases = (meta_schema.__class__,)
                if len(bases) > 1:
                    meta_bases += bases[1:]
            meta_schema_class = cast('XMLSchemaBase', super(XMLSchemaMeta, mcs).__new__(mcs, meta_schema_class_name, meta_bases, dict_))
            meta_schema_class.__qualname__ = meta_schema_class_name
            module = sys.modules[dict_['__module__']]
            setattr(module, meta_schema_class_name, meta_schema_class)
            meta_schema = meta_schema_class.create_meta_schema(meta_schema_file)
            dict_['meta_schema'] = meta_schema
        cls = super(XMLSchemaMeta, mcs).__new__(mcs, name, bases, dict_)
        if cls.XSD_VERSION not in ('1.0', '1.1'):
            raise XMLSchemaValueError(_("XSD_VERSION must be '1.0' or '1.1'"))
        return cls