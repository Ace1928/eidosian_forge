from typing import TYPE_CHECKING, Any, Optional, cast, Iterable, Union, Callable
from elementpath.etree import etree_tostring
from ..exceptions import XMLSchemaException, XMLSchemaWarning, XMLSchemaValueError
from ..aliases import ElementType, NamespacesType, SchemaElementType, ModelParticleType
from ..helpers import get_prefixed_qname, etree_getpath, is_etree_element
from ..translation import gettext as _
class XMLSchemaModelError(XMLSchemaValidatorError, ValueError):
    """
    Raised when a model error is found during the checking of a model group.

    :param group: the XSD model group.
    :param message: the error message.
    """

    def __init__(self, group: 'XsdGroup', message: str) -> None:
        super(XMLSchemaModelError, self).__init__(validator=group, message=message, elem=getattr(group, 'elem', None), source=getattr(group, 'source', None), namespaces=getattr(group, 'namespaces', None))