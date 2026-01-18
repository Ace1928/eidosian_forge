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
class XsdUnion(XsdSimpleType):
    """
    Class for 'union' definitions. A union definition has a member_types
    attribute that refers to a 'simpleType' definition.

    ..  <union
          id = ID
          memberTypes = List of QName
          {any attributes with non-schema namespace ...}>
          Content: (annotation?, simpleType*)
        </union>
    """
    member_types: Any = ()
    _ADMITTED_TYPES: Any = XsdSimpleType
    _ADMITTED_TAGS = {XSD_UNION}

    def __init__(self, elem: ElementType, schema: SchemaType, parent: Optional[XsdComponent], name: Optional[str]=None) -> None:
        super(XsdUnion, self).__init__(elem, schema, parent, name, facets=None)

    def __repr__(self) -> str:
        if self.name is None:
            return '%s(member_types=%r)' % (self.__class__.__name__, self.member_types)
        else:
            return '%s(name=%r)' % (self.__class__.__name__, self.prefixed_name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == 'elem' and value is not None and (value.tag != XSD_UNION):
            if value.tag == XSD_SIMPLE_TYPE:
                for child in value:
                    if child.tag == XSD_UNION:
                        super(XsdUnion, self).__setattr__(name, child)
                        return
            raise XMLSchemaValueError('a {0!r} definition required for {1!r}'.format(XSD_UNION, self))
        elif name == 'white_space':
            if not (value is None or value == 'collapse'):
                msg = _("wrong value %r for attribute 'white_space'")
                raise XMLSchemaValueError(msg % value)
            value = 'collapse'
        super(XsdUnion, self).__setattr__(name, value)

    def _parse(self) -> None:
        mt: Any
        member_types = []
        for child in self.elem:
            if child.tag != XSD_ANNOTATION and (not callable(child.tag)):
                mt = self.schema.simple_type_factory(child, parent=self)
                if isinstance(mt, XMLSchemaParseError):
                    self.parse_error(mt)
                else:
                    member_types.append(mt)
        if 'memberTypes' in self.elem.attrib:
            for name in self.elem.attrib['memberTypes'].split():
                try:
                    type_qname = self.schema.resolve_qname(name)
                except (KeyError, ValueError, RuntimeError) as err:
                    self.parse_error(err)
                    continue
                try:
                    mt = self.maps.lookup_type(type_qname)
                except KeyError:
                    self.parse_error(_('unknown type {!r}').format(type_qname))
                    mt = self.any_atomic_type
                except XMLSchemaParseError as err:
                    self.parse_error(err)
                    mt = self.any_atomic_type
                if isinstance(mt, tuple):
                    msg = _('circular definition found on xs:union type {!r}')
                    self.parse_error(msg.format(self.name))
                    continue
                elif not isinstance(mt, self._ADMITTED_TYPES):
                    msg = _('a {0!r} required, not {1!r}')
                    self.parse_error(msg.format(self._ADMITTED_TYPES, mt))
                    continue
                elif mt.final == '#all' or 'union' in mt.final:
                    msg = _("'final' value of the memberTypes %r forbids derivation by union")
                    self.parse_error(msg % member_types)
                member_types.append(mt)
        if not member_types:
            self.parse_error(_('missing xs:union type declarations'))
            self.member_types = [self.any_atomic_type]
        elif any((mt.name == XSD_ANY_ATOMIC_TYPE for mt in member_types)):
            msg = _('cannot use xs:anyAtomicType as base type of a user-defined type')
            self.parse_error(msg)
        else:
            self.member_types = member_types
            if all((not mt.allow_empty for mt in member_types)):
                self.allow_empty = False

    @property
    def variety(self) -> Optional[str]:
        return 'union'

    @property
    def admitted_facets(self) -> Set[str]:
        return XSD_10_UNION_FACETS if self.xsd_version == '1.0' else XSD_11_UNION_FACETS

    def is_atomic(self) -> bool:
        return all((mt.is_atomic() for mt in self.member_types))

    def is_list(self) -> bool:
        return all((mt.is_list() for mt in self.member_types))

    def is_key(self) -> bool:
        return any((mt.is_key() for mt in self.member_types))

    def is_union(self) -> bool:
        return True

    def is_dynamic_consistent(self, other: Any) -> bool:
        return other.name in {XSD_ANY_TYPE, XSD_ANY_SIMPLE_TYPE} or other.is_derived(self) or (isinstance(other, self.__class__) and any((mt1.is_derived(mt2) for mt1 in other.member_types for mt2 in self.member_types)))

    def iter_components(self, xsd_classes: ComponentClassType=None) -> Iterator[XsdComponent]:
        if xsd_classes is None or isinstance(self, xsd_classes):
            yield self
        for mt in filter(lambda x: x.parent is not None, self.member_types):
            yield from mt.iter_components(xsd_classes)

    def iter_decode(self, obj: AtomicValueType, validation: str='lax', patterns: Optional[XsdPatternFacets]=None, **kwargs: Any) -> IterDecodeType[DecodedValueType]:
        for member_type in self.member_types:
            for result in member_type.iter_decode(obj, validation='lax', **kwargs):
                if not isinstance(result, XMLSchemaValidationError):
                    if patterns and isinstance(obj, (str, bytes)):
                        try:
                            patterns(member_type.normalize(obj))
                        except XMLSchemaValidationError as err:
                            yield err
                    yield result
                    return
                break
        if isinstance(obj, bytes):
            obj = obj.decode('utf-8')
        if not isinstance(obj, str) or ' ' not in obj.strip():
            reason = _('invalid value {!r}').format(obj)
            yield XMLSchemaDecodeError(self, obj, self.member_types, reason)
            return
        items = []
        not_decodable = []
        for chunk in obj.split():
            for member_type in self.member_types:
                for result in member_type.iter_decode(chunk, validation='lax', **kwargs):
                    if isinstance(result, XMLSchemaValidationError):
                        break
                    else:
                        items.append(result)
                else:
                    break
            else:
                if validation != 'skip':
                    not_decodable.append(chunk)
                else:
                    items.append(str(chunk))
        if not_decodable:
            reason = _('no type suitable for decoding the values %r') % not_decodable
            yield XMLSchemaDecodeError(self, obj, self.member_types, reason)
        yield (items if len(items) > 1 else items[0] if items else None)

    def iter_encode(self, obj: Any, validation: str='lax', **kwargs: Any) -> IterEncodeType[EncodedValueType]:
        for member_type in self.member_types:
            for result in member_type.iter_encode(obj, validation='lax', **kwargs):
                if result is not None and (not isinstance(result, XMLSchemaValidationError)):
                    yield result
                    return
                elif validation == 'strict':
                    break
        if hasattr(obj, '__iter__') and (not isinstance(obj, (str, bytes))):
            for member_type in self.member_types:
                results = []
                for item in obj:
                    for result in member_type.iter_encode(item, validation='lax', **kwargs):
                        if result is not None and (not isinstance(result, XMLSchemaValidationError)):
                            results.append(result)
                            break
                        elif validation == 'strict':
                            break
                if len(results) == len(obj):
                    yield results
                    break
        if validation != 'skip':
            reason = _('no type suitable for encoding the object')
            yield XMLSchemaEncodeError(self, obj, self.member_types, reason)
            yield None
        else:
            yield str(obj)