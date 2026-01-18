import re
from typing import TYPE_CHECKING, cast, Any, Dict, Generic, List, Iterator, Optional, \
from xml.etree import ElementTree
from elementpath import select
from elementpath.etree import is_etree_element, etree_tostring
from ..exceptions import XMLSchemaValueError, XMLSchemaTypeError
from ..names import XSD_ANNOTATION, XSD_APPINFO, XSD_DOCUMENTATION, \
from ..aliases import ElementType, NamespacesType, SchemaType, BaseXsdType, \
from ..translation import gettext as _
from ..helpers import get_qname, local_name, get_prefixed_qname
from ..resources import XMLResource
from .exceptions import XMLSchemaParseError, XMLSchemaValidationError
def _parse_xpath_default_namespace(self, elem: ElementType) -> str:
    """
        Parse XSD 1.1 xpathDefaultNamespace attribute for schema, alternative, assert, assertion
        and selector declarations, checking if the value is conforming to the specification. In
        case the attribute is missing or for wrong attribute values defaults to ''.
        """
    try:
        value = elem.attrib['xpathDefaultNamespace']
    except KeyError:
        return ''
    value = value.strip()
    if value == '##local':
        return ''
    elif value == '##defaultNamespace':
        default_namespace = getattr(self, 'default_namespace', None)
        return default_namespace if isinstance(default_namespace, str) else ''
    elif value == '##targetNamespace':
        target_namespace = getattr(self, 'target_namespace', '')
        return target_namespace if isinstance(target_namespace, str) else ''
    elif len(value.split()) == 1:
        return value
    else:
        admitted_values = ('##defaultNamespace', '##targetNamespace', '##local')
        msg = _("wrong value {0!r} for 'xpathDefaultNamespace' attribute, can be (anyURI | {1}).")
        self.parse_error(msg.format(value, ' | '.join(admitted_values)), elem)
        return ''