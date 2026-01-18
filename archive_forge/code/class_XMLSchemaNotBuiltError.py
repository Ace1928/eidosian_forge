from typing import TYPE_CHECKING, Any, Optional, cast, Iterable, Union, Callable
from elementpath.etree import etree_tostring
from ..exceptions import XMLSchemaException, XMLSchemaWarning, XMLSchemaValueError
from ..aliases import ElementType, NamespacesType, SchemaElementType, ModelParticleType
from ..helpers import get_prefixed_qname, etree_getpath, is_etree_element
from ..translation import gettext as _
class XMLSchemaNotBuiltError(XMLSchemaValidatorError, RuntimeError):
    """
    Raised when there is an improper usage attempt of a not built XSD validator.

    :param validator: the XSD validator.
    :param message: the error message.
    """

    def __init__(self, validator: 'XsdValidator', message: str) -> None:
        super(XMLSchemaNotBuiltError, self).__init__(validator=validator, message=message, elem=getattr(validator, 'elem', None), source=getattr(validator, 'source', None), namespaces=getattr(validator, 'namespaces', None))