from typing import TYPE_CHECKING, Any, Optional, cast, Iterable, Union, Callable
from elementpath.etree import etree_tostring
from ..exceptions import XMLSchemaException, XMLSchemaWarning, XMLSchemaValueError
from ..aliases import ElementType, NamespacesType, SchemaElementType, ModelParticleType
from ..helpers import get_prefixed_qname, etree_getpath, is_etree_element
from ..translation import gettext as _
class XMLSchemaValidatorError(XMLSchemaException):
    """
    Base class for XSD validator errors.

    :param validator: the XSD validator.
    :param message: the error message.
    :param elem: the element that contains the error.
    :param source: the XML resource that contains the error.
    :param namespaces: is an optional mapping from namespace prefix to URI.
    """
    _path: Optional[str]

    def __init__(self, validator: ValidatorType, message: str, elem: Optional[ElementType]=None, source: Optional['XMLResource']=None, namespaces: Optional[NamespacesType]=None) -> None:
        self._path = None
        self.validator = validator
        self.message = message[:-1] if message[-1] in ('.', ':') else message
        self.namespaces = namespaces
        self.source = source
        self.elem = elem

    def __str__(self) -> str:
        if self.elem is None:
            return self.message
        msg = ['%s:\n' % self.message]
        elem_as_string = cast(str, etree_tostring(self.elem, self.namespaces, '  ', 20))
        msg.append('Schema:\n\n%s\n' % elem_as_string)
        path = self.path
        if path is not None:
            msg.append('Path: %s\n' % path)
        if self.schema_url is not None:
            msg.append('Schema URL: %s\n' % self.schema_url)
            if self.origin_url not in (None, self.schema_url):
                msg.append('Origin URL: %s\n' % self.origin_url)
        return '\n'.join(msg)

    @property
    def msg(self) -> str:
        return self.__str__()

    def __setattr__(self, name: str, value: Any) -> None:
        if name == 'elem' and value is not None:
            if not is_etree_element(value):
                raise XMLSchemaValueError("'elem' attribute requires an Element, not %r." % type(value))
            if self.source is not None:
                self._path = etree_getpath(elem=value, root=self.source.root, namespaces=self.namespaces, relative=False, add_position=True)
                if self.source.is_lazy():
                    value = None
        super(XMLSchemaValidatorError, self).__setattr__(name, value)

    @property
    def sourceline(self) -> Any:
        """XML element *sourceline* if available (lxml Element) and *elem* is set."""
        return getattr(self.elem, 'sourceline', None)

    @property
    def root(self) -> Optional[ElementType]:
        """The XML resource root element if *source* is set."""
        try:
            return self.source.root
        except AttributeError:
            return None

    @property
    def schema_url(self) -> Optional[str]:
        """The schema URL, if available and the *validator* is an XSD component."""
        url: Optional[str]
        try:
            url = self.validator.schema.source.url
        except AttributeError:
            return None
        else:
            return url

    @property
    def origin_url(self) -> Optional[str]:
        """The origin schema URL, if available and the *validator* is an XSD component."""
        url: Optional[str]
        try:
            url = self.validator.maps.validator.source.url
        except AttributeError:
            return None
        else:
            return url

    @property
    def path(self) -> Optional[str]:
        """The XPath of the element, if it's not `None` and the XML resource is set."""
        if self.elem is None or self.source is None:
            return self._path
        return etree_getpath(elem=self.elem, root=self.source.root, namespaces=self.namespaces, relative=False, add_position=True)