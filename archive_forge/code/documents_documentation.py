import json
import copy
from io import IOBase, TextIOBase
from typing import Any, Dict, List, Optional, Type, Union, Tuple, \
from elementpath.etree import ElementTree, etree_tostring
from .exceptions import XMLSchemaTypeError, XMLSchemaValueError, XMLResourceError
from .names import XSD_NAMESPACE, XSI_TYPE, XSD_SCHEMA
from .aliases import ElementType, XMLSourceType, NamespacesType, LocationsType, \
from .helpers import get_extended_qname, is_etree_document
from .resources import fetch_schema_locations, XMLResource
from .validators import XMLSchema10, XMLSchemaBase, XMLSchemaValidationError
Serialize an XML resource to a file. Cannot be used with lazy resources.