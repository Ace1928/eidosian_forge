from decimal import DecimalException
from typing import cast, Any, Callable, Dict, Iterator, List, \
from xml.etree import ElementTree
from ..aliases import ElementType, AtomicValueType, ComponentClassType, \
from ..exceptions import XMLSchemaTypeError, XMLSchemaValueError
from ..names import XSD_NAMESPACE, XSD_ANY_TYPE, XSD_SIMPLE_TYPE, XSD_PATTERN, \
from ..translation import gettext as _
from ..helpers import local_name
from .exceptions import XMLSchemaValidationError, XMLSchemaEncodeError, \
from .xsdbase import XsdComponent, XsdType, ValidationMixin
from .facets import XsdFacet, XsdWhiteSpaceFacet, XsdPatternFacets, \
class XsdAtomicBuiltin(XsdAtomic):
    """
    Class for defining XML Schema built-in simpleType atomic datatypes. An instance
    contains a Python's type transformation and a list of validator functions. The
    'base_type' is not used for validation, but only for reference to the XML Schema
    restriction hierarchy.

    Type conversion methods:
      - to_python(value): Decoding from XML
      - from_python(value): Encoding to XML
    """

    def __init__(self, elem: ElementType, schema: SchemaType, name: str, python_type: Type[Any], base_type: Optional['XsdAtomicBuiltin']=None, admitted_facets: Optional[Set[str]]=None, facets: Optional[Dict[Optional[str], FacetsValueType]]=None, to_python: Any=None, from_python: Any=None) -> None:
        """
        :param name: the XSD type's qualified name.
        :param python_type: the correspondent Python's type. If a tuple of types         is provided uses the first and consider the others as compatible types.
        :param base_type: the reference base type, None if it's a primitive type.
        :param admitted_facets: admitted facets tags for type (required for primitive types).
        :param facets: optional facets validators.
        :param to_python: optional decode function.
        :param from_python: optional encode function.
        """
        if isinstance(python_type, tuple):
            self.instance_types, python_type = (python_type, python_type[0])
        else:
            self.instance_types = python_type
        if not callable(python_type):
            raise XMLSchemaTypeError('%r object is not callable' % python_type.__class__)
        if base_type is None and (not admitted_facets) and (name != XSD_ERROR):
            raise XMLSchemaValueError("argument 'admitted_facets' must be a not empty set of a primitive type")
        self._admitted_facets = admitted_facets
        super(XsdAtomicBuiltin, self).__init__(elem, schema, None, name, facets, base_type)
        self.python_type = python_type
        self.to_python = to_python if to_python is not None else python_type
        self.from_python = from_python if from_python is not None else str

    def __repr__(self) -> str:
        return '%s(name=%r)' % (self.__class__.__name__, self.prefixed_name)

    @property
    def admitted_facets(self) -> Set[str]:
        return self._admitted_facets or self.primitive_type.admitted_facets

    def iter_decode(self, obj: Union[str, bytes], validation: str='lax', **kwargs: Any) -> IterDecodeType[DecodedValueType]:
        if isinstance(obj, (str, bytes)):
            obj = self.normalize(obj)
        elif obj is not None and (not isinstance(obj, self.instance_types)):
            reason = _('value is not an instance of {!r}').format(self.instance_types)
            yield XMLSchemaDecodeError(self, obj, self.to_python, reason)
        if validation == 'skip':
            try:
                yield self.to_python(obj)
            except (ValueError, DecimalException):
                yield str(obj)
            return
        if self.patterns is not None:
            try:
                self.patterns(obj)
            except XMLSchemaValidationError as err:
                yield err
        try:
            result = self.to_python(obj)
        except (ValueError, DecimalException) as err:
            yield XMLSchemaDecodeError(self, obj, self.to_python, reason=str(err))
            yield None
            return
        except TypeError:
            reason = _('invalid value {!r}').format(obj)
            yield self.validation_error(validation, error=reason, obj=obj)
            yield None
            return
        for validator in self.validators:
            try:
                validator(result)
            except XMLSchemaValidationError as err:
                yield err
        if self.name not in {XSD_QNAME, XSD_IDREF, XSD_ID}:
            pass
        elif self.name == XSD_QNAME:
            if ':' in obj:
                try:
                    prefix, name = obj.split(':')
                except ValueError:
                    pass
                else:
                    try:
                        result = f'{{{kwargs['namespaces'][prefix]}}}{name}'
                    except (TypeError, KeyError):
                        try:
                            if kwargs['source'].namespace != XSD_NAMESPACE:
                                reason = _('unmapped prefix %r in a QName') % prefix
                                yield self.validation_error(validation, error=reason, obj=obj)
                        except KeyError:
                            pass
            else:
                try:
                    default_namespace = kwargs['namespaces']['']
                except (TypeError, KeyError):
                    pass
                else:
                    if default_namespace:
                        result = f'{{{default_namespace}}}{obj}'
        elif self.name == XSD_IDREF:
            try:
                id_map = kwargs['id_map']
            except KeyError:
                pass
            else:
                if obj not in id_map:
                    id_map[obj] = 0
        elif kwargs.get('level') != 0:
            try:
                id_map = kwargs['id_map']
            except KeyError:
                pass
            else:
                try:
                    id_list = kwargs['id_list']
                except KeyError:
                    if not id_map[obj]:
                        id_map[obj] = 1
                    else:
                        reason = _('duplicated xs:ID value {!r}').format(obj)
                        yield self.validation_error(validation, error=reason, obj=obj)
                else:
                    if not id_map[obj]:
                        id_map[obj] = 1
                        id_list.append(obj)
                        if len(id_list) > 1 and self.xsd_version == '1.0':
                            reason = _('no more than one attribute of type ID should be present in an element')
                            yield self.validation_error(validation, reason, obj, **kwargs)
                    elif obj not in id_list or self.xsd_version == '1.0':
                        reason = _('duplicated xs:ID value {!r}').format(obj)
                        yield self.validation_error(validation, error=reason, obj=obj)
        yield result

    def iter_encode(self, obj: Any, validation: str='lax', **kwargs: Any) -> IterEncodeType[EncodedValueType]:
        if isinstance(obj, (str, bytes)):
            obj = self.normalize(obj)
        if validation == 'skip':
            try:
                yield self.from_python(obj)
            except ValueError:
                yield str(obj)
            return
        elif isinstance(obj, bool):
            types_: Any = self.instance_types
            if types_ is not bool or (isinstance(types_, tuple) and bool in types_):
                reason = _('boolean value {0!r} requires a {1!r} decoder').format(obj, bool)
                yield XMLSchemaEncodeError(self, obj, self.from_python, reason)
                obj = self.python_type(obj)
        elif not isinstance(obj, self.instance_types):
            reason = _('{0!r} is not an instance of {1!r}').format(obj, self.instance_types)
            yield XMLSchemaEncodeError(self, obj, self.from_python, reason)
            try:
                value = self.python_type(obj)
                if value != obj and (not isinstance(value, str)) and (not isinstance(obj, (str, bytes))):
                    raise ValueError()
                obj = value
            except (ValueError, TypeError) as err:
                yield XMLSchemaEncodeError(self, obj, self.from_python, reason=str(err))
                yield None
                return
            else:
                if value == obj or str(value) == str(obj):
                    obj = value
                else:
                    reason = _('invalid value {!r}').format(obj)
                    yield XMLSchemaEncodeError(self, obj, self.from_python, reason)
                    yield None
                    return
        for validator in self.validators:
            try:
                validator(obj)
            except XMLSchemaValidationError as err:
                yield err
        try:
            text = self.from_python(obj)
        except ValueError as err:
            yield XMLSchemaEncodeError(self, obj, self.from_python, reason=str(err))
            yield None
        else:
            if self.patterns is not None:
                try:
                    self.patterns(text)
                except XMLSchemaValidationError as err:
                    yield err
            yield text