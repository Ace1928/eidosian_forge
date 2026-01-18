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
class XsdValidator:
    """
    Common base class for XML Schema validator, that represents a PSVI (Post Schema Validation
    Infoset) information item. A concrete XSD validator have to report its validity collecting
    building errors and implementing the properties.

    :param validation: defines the XSD validation mode to use for build the validator,     its value can be 'strict', 'lax' or 'skip'. Strict mode is the default.
    :type validation: str

    :ivar validation: XSD validation mode.
    :vartype validation: str
    :ivar errors: XSD validator building errors.
    :vartype errors: list
    """
    elem: Optional[ElementTree.Element] = None
    namespaces: Any = None
    errors: List[XMLSchemaParseError]

    def __init__(self, validation: str='strict') -> None:
        self.validation = validation
        self.errors = []

    @property
    def built(self) -> bool:
        """
        Property that is ``True`` if XSD validator has been fully parsed and built,
        ``False`` otherwise. For schemas the property is checked on all global
        components. For XSD components check only the building of local subcomponents.
        """
        raise NotImplementedError()

    @property
    def validation_attempted(self) -> str:
        """
        Property that returns the *validation status* of the XSD validator.
        It can be 'full', 'partial' or 'none'.

        | https://www.w3.org/TR/xmlschema-1/#e-validation_attempted
        | https://www.w3.org/TR/2012/REC-xmlschema11-1-20120405/#e-validation_attempted
        """
        raise NotImplementedError()

    @property
    def validity(self) -> str:
        """
        Property that returns the XSD validator's validity.
        It can be ‘valid’, ‘invalid’ or ‘notKnown’.

        | https://www.w3.org/TR/xmlschema-1/#e-validity
        | https://www.w3.org/TR/2012/REC-xmlschema11-1-20120405/#e-validity
        """
        if self.validation == 'skip':
            return 'notKnown'
        elif self.errors or any((comp.errors for comp in self.iter_components())):
            return 'invalid'
        elif self.built:
            return 'valid'
        else:
            return 'notKnown'

    def iter_components(self, xsd_classes: ComponentClassType=None) -> Iterator[Union['XsdComponent', SchemaType, 'XsdGlobals']]:
        """
        Creates an iterator for traversing all XSD components of the validator.

        :param xsd_classes: returns only a specific class/classes of components,         otherwise returns all components.
        """
        raise NotImplementedError()

    @property
    def all_errors(self) -> List[XMLSchemaParseError]:
        """
        A list with all the building errors of the XSD validator and its components.
        """
        errors = []
        for comp in self.iter_components():
            if comp.errors:
                errors.extend(comp.errors)
        return errors

    def copy(self) -> 'XsdValidator':
        validator: 'XsdValidator' = object.__new__(self.__class__)
        validator.__dict__.update(self.__dict__)
        validator.errors = self.errors[:]
        return validator
    __copy__ = copy

    def parse_error(self, error: Union[str, Exception], elem: Optional[ElementType]=None, validation: Optional[str]=None) -> None:
        """
        Helper method for registering parse errors. Does nothing if validation mode is 'skip'.
        Il validation mode is 'lax' collects the error, otherwise raise the error.

        :param error: can be a parse error or an error message.
        :param elem: the Element instance related to the error, for default uses the 'elem'         attribute of the validator, if it's present.
        :param validation: overrides the default validation mode of the validator.
        """
        if validation is not None:
            check_validation_mode(validation)
        else:
            validation = self.validation
        if validation == 'skip':
            return
        elif elem is None:
            elem = self.elem
        elif not is_etree_element(elem):
            msg = "the argument 'elem' must be an Element instance, not {!r}."
            raise XMLSchemaTypeError(msg.format(elem))
        if isinstance(error, XMLSchemaParseError):
            error.validator = self
            error.namespaces = getattr(self, 'namespaces', None)
            error.elem = elem
            error.source = getattr(self, 'source', None)
        elif isinstance(error, Exception):
            message = str(error).strip()
            if message[0] in '\'"' and message[0] == message[-1]:
                message = message.strip('\'"')
            error = XMLSchemaParseError(self, message, elem)
        elif isinstance(error, str):
            error = XMLSchemaParseError(self, error, elem)
        else:
            msg = "'error' argument must be an exception or a string, not {!r}."
            raise XMLSchemaTypeError(msg.format(error))
        if validation == 'lax':
            self.errors.append(error)
        else:
            raise error

    def validation_error(self, validation: str, error: Union[str, Exception], obj: Any=None, source: Optional[XMLResource]=None, namespaces: Optional[NamespacesType]=None, **_kwargs: Any) -> XMLSchemaValidationError:
        """
        Helper method for generating and updating validation errors. If validation
        mode is 'lax' or 'skip' returns the error, otherwise raises the error.

        :param validation: an error-compatible validation mode: can be 'lax' or 'strict'.
        :param error: an error instance or the detailed reason of failed validation.
        :param obj: the instance related to the error.
        :param source: the XML resource related to the validation process.
        :param namespaces: is an optional mapping from namespace prefix to URI.
        :param _kwargs: keyword arguments of the validation process that are not used.
        """
        check_validation_mode(validation)
        if isinstance(error, XMLSchemaValidationError):
            if error.namespaces is None and namespaces is not None:
                error.namespaces = namespaces
            if error.source is None and source is not None:
                error.source = source
            if error.obj is None and obj is not None:
                error.obj = obj
            if error.elem is None and obj is not None and is_etree_element(obj):
                error.elem = obj
                if is_etree_element(error.obj) and obj.tag == error.obj.tag:
                    error.obj = obj
        elif isinstance(error, Exception):
            error = XMLSchemaValidationError(self, obj, str(error), source, namespaces)
        else:
            error = XMLSchemaValidationError(self, obj, error, source, namespaces)
        if validation == 'strict' and error.elem is not None:
            raise error
        return error

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