from typing import TYPE_CHECKING, Any, Optional, cast, Iterable, Union, Callable
from elementpath.etree import etree_tostring
from ..exceptions import XMLSchemaException, XMLSchemaWarning, XMLSchemaValueError
from ..aliases import ElementType, NamespacesType, SchemaElementType, ModelParticleType
from ..helpers import get_prefixed_qname, etree_getpath, is_etree_element
from ..translation import gettext as _
class XMLSchemaDecodeError(XMLSchemaValidationError):
    """
    Raised when an XML data string is not decodable to a Python object.

    :param validator: the XSD validator.
    :param obj: the not validated XML data.
    :param decoder: the XML data decoder.
    :param reason: the detailed reason of failed validation.
    :param source: the XML resource that contains the error.
    :param namespaces: is an optional mapping from namespace prefix to URI.
    """
    message = 'failed decoding {!r} with {!r}.\n'

    def __init__(self, validator: Union['XsdValidator', Callable[[Any], None]], obj: Any, decoder: Any, reason: Optional[str]=None, source: Optional['XMLResource']=None, namespaces: Optional[NamespacesType]=None) -> None:
        super(XMLSchemaDecodeError, self).__init__(validator, obj, reason, source, namespaces)
        self.decoder = decoder