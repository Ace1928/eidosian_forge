from typing import TYPE_CHECKING, Any, Optional, cast, Iterable, Union, Callable
from elementpath.etree import etree_tostring
from ..exceptions import XMLSchemaException, XMLSchemaWarning, XMLSchemaValueError
from ..aliases import ElementType, NamespacesType, SchemaElementType, ModelParticleType
from ..helpers import get_prefixed_qname, etree_getpath, is_etree_element
from ..translation import gettext as _
class XMLSchemaValidationError(XMLSchemaValidatorError, ValueError):
    """
    Raised when the XML data is not validated with the XSD component or schema.
    It's used by decoding and encoding methods. Encoding validation errors do
    not include XML data element and source, so the error is limited to a message
    containing object representation and a reason.

    :param validator: the XSD validator.
    :param obj: the not validated XML data.
    :param reason: the detailed reason of failed validation.
    :param source: the XML resource that contains the error.
    :param namespaces: is an optional mapping from namespace prefix to URI.
    """

    def __init__(self, validator: ValidatorType, obj: Any, reason: Optional[str]=None, source: Optional['XMLResource']=None, namespaces: Optional[NamespacesType]=None) -> None:
        if not isinstance(obj, str):
            _obj = obj
        else:
            _obj = obj.encode('ascii', 'xmlcharrefreplace').decode('utf-8')
        super(XMLSchemaValidationError, self).__init__(validator=validator, message='failed validating {!r} with {!r}'.format(_obj, validator), elem=obj if is_etree_element(obj) else None, source=source, namespaces=namespaces)
        self.obj = obj
        self.reason = reason

    def __repr__(self) -> str:
        return '%s(reason=%r)' % (self.__class__.__name__, self.reason)

    def __str__(self) -> str:
        msg = ['%s:\n' % self.message]
        if self.reason is not None:
            msg.append('Reason: %s\n' % self.reason)
        if hasattr(self.validator, 'tostring'):
            chunk = self.validator.tostring('  ', 20)
            msg.append('Schema:\n\n%s\n' % chunk)
        if self.elem is not None and is_etree_element(self.elem):
            try:
                elem_as_string = cast(str, etree_tostring(self.elem, self.namespaces, '  ', 20))
            except (ValueError, TypeError):
                elem_as_string = repr(self.elem)
            if hasattr(self.elem, 'sourceline'):
                line = getattr(self.elem, 'sourceline')
                msg.append('Instance (line %r):\n\n%s\n' % (line, elem_as_string))
            else:
                msg.append('Instance:\n\n%s\n' % elem_as_string)
        if self.path is not None:
            msg.append('Path: %s\n' % self.path)
        if len(msg) == 1:
            return msg[0][:-2]
        return '\n'.join(msg)