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

        Creates and iterator for all XSD subelements matching the path.

        :param path: an XPath expression that considers the data element as the root.
        :param namespaces: is an optional mapping from namespace prefix to full name.
        :return: an iterable yielding all matching data elements in document order.
        