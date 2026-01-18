import sys
from abc import abstractmethod
from typing import cast, overload, Any, Dict, Iterator, List, Optional, \
import re
from elementpath import XPath2Parser, XPathSchemaContext, \
from .exceptions import XMLSchemaValueError, XMLSchemaTypeError
from .names import XSD_NAMESPACE
from .aliases import NamespacesType, SchemaType, BaseXsdType, XPathElementType
from .helpers import get_qname, local_name, get_prefixed_qname
def _get_xpath_namespaces(self, namespaces: Optional[NamespacesType]=None) -> Dict[str, str]:
    """
        Returns a dictionary with namespaces for XPath selection.

        :param namespaces: an optional map from namespace prefix to namespace URI.         If this argument is not provided the schema's namespaces are used.
        """
    xpath_namespaces: Dict[str, str] = XPath2Parser.DEFAULT_NAMESPACES.copy()
    if namespaces is None:
        xpath_namespaces.update(self.namespaces)
    else:
        xpath_namespaces.update(namespaces)
    return xpath_namespaces