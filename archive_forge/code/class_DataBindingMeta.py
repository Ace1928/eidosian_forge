import re
from abc import ABCMeta
from copy import copy
from itertools import count
from typing import TYPE_CHECKING, cast, overload, Any, Dict, List, Iterator, \
from elementpath import XPathContext, XPath2Parser, build_node_tree, protocols
from elementpath.etree import etree_tostring
from .exceptions import XMLSchemaAttributeError, XMLSchemaTypeError, XMLSchemaValueError
from .aliases import ElementType, XMLSourceType, NamespacesType, BaseXsdType, DecodeType
from .helpers import get_namespace, get_prefixed_qname, local_name, raw_xml_encode
from .converters import ElementData, XMLSchemaConverter
from .resources import XMLResource
from . import validators
class DataBindingMeta(ABCMeta):
    """Metaclass for creating classes with bindings to XSD elements."""
    xsd_element: 'XsdElement'

    def __new__(mcs, name: str, bases: Tuple[Type[Any], ...], attrs: Dict[str, Any]) -> 'DataBindingMeta':
        try:
            xsd_element = attrs['xsd_element']
        except KeyError:
            msg = "attribute 'xsd_element' is required for an XSD data binding class"
            raise XMLSchemaAttributeError(msg) from None
        if not isinstance(xsd_element, validators.XsdElement):
            raise XMLSchemaTypeError('{!r} is not an XSD element'.format(xsd_element))
        attrs['__module__'] = None
        return super(DataBindingMeta, mcs).__new__(mcs, name, bases, attrs)

    def __init__(cls, name: str, bases: Tuple[Type[Any], ...], attrs: Dict[str, Any]) -> None:
        super(DataBindingMeta, cls).__init__(name, bases, attrs)
        cls.xsd_version = cls.xsd_element.xsd_version
        cls.namespace = cls.xsd_element.target_namespace

    def fromsource(cls, source: Union[XMLSourceType, XMLResource], allow: str='all', defuse: str='remote', timeout: int=300, **kwargs: Any) -> DecodeType[Any]:
        if not isinstance(source, XMLResource):
            source = XMLResource(source, allow=allow, defuse=defuse, timeout=timeout)
        if 'converter' not in kwargs:
            kwargs['converter'] = DataBindingConverter
        return cls.xsd_element.schema.decode(source, **kwargs)