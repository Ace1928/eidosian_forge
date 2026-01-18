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
class XsdAtomicRestriction(XsdAtomic):
    """
    Class for XSD 1.0 atomic simpleType and complexType's simpleContent restrictions.

    ..  <restriction
          base = QName
          id = ID
          {any attributes with non-schema namespace . . .}>
          Content: (annotation?, (simpleType?, (minExclusive | minInclusive | maxExclusive |
          maxInclusive | totalDigits | fractionDigits | length | minLength | maxLength |
          enumeration | whiteSpace | pattern)*))
        </restriction>
    """
    parent: 'XsdSimpleType'
    base_type: BaseXsdType
    derivation = 'restriction'
    _FACETS_BUILDERS = XSD_10_FACETS_BUILDERS
    _CONTENT_TAIL_TAGS = {XSD_ATTRIBUTE, XSD_ATTRIBUTE_GROUP, XSD_ANY_ATTRIBUTE}

    def __setattr__(self, name: str, value: Any) -> None:
        if name == 'elem' and value is not None:
            if self.name != XSD_ANY_ATOMIC_TYPE and value.tag != XSD_RESTRICTION:
                if not (value.tag == XSD_SIMPLE_TYPE and value.get('name') is not None):
                    raise XMLSchemaValueError('an xs:restriction definition required for %r.' % self)
        super(XsdAtomicRestriction, self).__setattr__(name, value)

    def _parse(self) -> None:
        elem = self.elem
        if elem.get('name') == XSD_ANY_ATOMIC_TYPE:
            return
        elif elem.tag == XSD_SIMPLE_TYPE and elem.get('name') is not None:
            elem = cast(ElementType, self._parse_child_component(elem))
        if self.name is not None and self.parent is not None:
            msg = _("'name' attribute in a local simpleType definition")
            self.parse_error(msg)
        base_type: Any = None
        facets: Any = {}
        has_attributes = False
        has_simple_type_child = False
        if 'base' in elem.attrib:
            try:
                base_qname = self.schema.resolve_qname(elem.attrib['base'])
            except (KeyError, ValueError, RuntimeError) as err:
                self.parse_error(err)
                base_type = self.any_atomic_type
            else:
                if base_qname == self.name:
                    if self.redefine is None:
                        msg = _('wrong definition with self-reference')
                        self.parse_error(msg)
                        base_type = self.any_atomic_type
                    else:
                        base_type = self.base_type
                else:
                    if self.redefine is not None:
                        msg = _('wrong redefinition without self-reference')
                        self.parse_error(msg)
                    try:
                        base_type = self.maps.lookup_type(base_qname)
                    except KeyError:
                        self.parse_error(_('unknown type {!r}').format(elem.attrib['base']))
                        base_type = self.any_atomic_type
                    except XMLSchemaParseError as err:
                        self.parse_error(err)
                        base_type = self.any_atomic_type
                    else:
                        if isinstance(base_type, tuple):
                            msg = _('circular definition found between {0!r} and {1!r}')
                            self.parse_error(msg.format(self, base_qname))
                            base_type = self.any_atomic_type
            if base_type.is_simple() and base_type.name == XSD_ANY_SIMPLE_TYPE:
                msg = _('wrong base type %r, an atomic type required')
                self.parse_error(msg % XSD_ANY_SIMPLE_TYPE)
            elif base_type.is_complex():
                if base_type.mixed and base_type.is_emptiable():
                    child = self._parse_child_component(elem, strict=False)
                    if child is None:
                        msg = _('an xs:simpleType definition expected')
                        self.parse_error(msg)
                    elif child.tag != XSD_SIMPLE_TYPE:
                        self.parse_error(_('when a complexType with simpleContent restricts a complexType with mixed and with emptiable content then a simpleType child declaration is required'))
                elif self.parent is None or self.parent.is_simple():
                    msg = _('simpleType restriction of %r is not allowed')
                    self.parse_error(msg % base_type)
        for child in elem:
            if child.tag == XSD_ANNOTATION or callable(child.tag):
                continue
            elif child.tag in self._CONTENT_TAIL_TAGS:
                has_attributes = True
            elif has_attributes:
                msg = _('unexpected tag after attribute declarations')
                self.parse_error(msg)
            elif child.tag == XSD_SIMPLE_TYPE:
                if has_simple_type_child:
                    msg = _('duplicated simpleType declaration')
                    self.parse_error(msg)
                if base_type is None:
                    try:
                        base_type = self.schema.simple_type_factory(child, parent=self)
                    except XMLSchemaParseError as err:
                        self.parse_error(err, child)
                        base_type = self.any_simple_type
                elif base_type.is_complex():
                    if base_type.admit_simple_restriction():
                        base_type = self.schema.xsd_complex_type_class(elem=elem, schema=self.schema, parent=self, content=self.schema.simple_type_factory(child, parent=self), attributes=base_type.attributes, mixed=base_type.mixed, block=base_type.block, final=base_type.final)
                elif 'base' in elem.attrib:
                    msg = _("restriction with 'base' attribute and simpleType declaration")
                    self.parse_error(msg)
                has_simple_type_child = True
            else:
                try:
                    facet_class = self._FACETS_BUILDERS[child.tag]
                except KeyError:
                    self.parse_error(_('unexpected tag %r in restriction') % child.tag)
                    continue
                if child.tag not in facets:
                    facets[child.tag] = facet_class(child, self.schema, self, base_type)
                elif child.tag not in MULTIPLE_FACETS:
                    msg = _('multiple %r constraint facet')
                    self.parse_error(msg % local_name(child.tag))
                elif child.tag != XSD_ASSERTION:
                    facets[child.tag].append(child)
                else:
                    assertion = facet_class(child, self.schema, self, base_type)
                    try:
                        facets[child.tag].append(assertion)
                    except AttributeError:
                        facets[child.tag] = [facets[child.tag], assertion]
        if base_type is None:
            self.parse_error(_('missing base type in restriction'))
        elif base_type.final == '#all' or 'restriction' in base_type.final:
            msg = _("'final' value of the baseType %r forbids derivation by restriction")
            self.parse_error(msg % base_type)
        if base_type is self.any_atomic_type:
            msg = _('cannot use xs:anyAtomicType as base type of a user-defined type')
            self.parse_error(msg)
        self.base_type = base_type
        self.facets = facets

    @property
    def variety(self) -> Optional[str]:
        return cast(Optional[str], getattr(self.base_type, 'variety', None))

    def iter_components(self, xsd_classes: ComponentClassType=None) -> Iterator[XsdComponent]:
        if xsd_classes is None:
            yield self
            for facet in self.facets.values():
                if isinstance(facet, list):
                    yield from facet
                elif isinstance(facet, XsdFacet):
                    yield facet
        else:
            if isinstance(self, xsd_classes):
                yield self
            if issubclass(XsdFacet, xsd_classes):
                for facet in self.facets.values():
                    if isinstance(facet, list):
                        yield from facet
                    elif isinstance(facet, XsdFacet):
                        yield facet
        if self.base_type.parent is not None:
            yield from self.base_type.iter_components(xsd_classes)

    def iter_decode(self, obj: AtomicValueType, validation: str='lax', **kwargs: Any) -> IterDecodeType[DecodedValueType]:
        if isinstance(obj, (str, bytes)):
            obj = self.normalize(obj)
            if self.patterns:
                if not isinstance(self.primitive_type, XsdUnion):
                    try:
                        self.patterns(obj)
                    except XMLSchemaValidationError as err:
                        yield err
                elif 'patterns' not in kwargs:
                    kwargs['patterns'] = self.patterns
        base_type: Any
        if isinstance(self.base_type, XsdSimpleType):
            base_type = self.base_type
        elif self.base_type.has_simple_content():
            base_type = self.base_type.content
        elif self.base_type.mixed:
            yield obj
            return
        else:
            msg = _('wrong base type %r: a simpleType or a complexType with simple or mixed content required')
            raise XMLSchemaValueError(msg % self.base_type)
        for result in base_type.iter_decode(obj, validation, **kwargs):
            if isinstance(result, XMLSchemaValidationError):
                yield result
            else:
                if result is not None:
                    for validator in self.validators:
                        try:
                            validator(result)
                        except XMLSchemaValidationError as err:
                            yield err
                yield result
                return

    def iter_encode(self, obj: Any, validation: str='lax', **kwargs: Any) -> IterEncodeType[EncodedValueType]:
        base_type: Any
        if self.is_list():
            if not hasattr(obj, '__iter__') or isinstance(obj, (str, bytes)):
                obj = [] if obj is None or obj == '' else [obj]
            base_type = self.base_type
        else:
            if isinstance(obj, (str, bytes)):
                obj = self.normalize(obj)
            if isinstance(self.base_type, XsdSimpleType):
                base_type = self.base_type
            elif self.base_type.has_simple_content():
                base_type = self.base_type.content
            elif self.base_type.mixed:
                yield str(obj)
                return
            else:
                msg = _('wrong base type %r: a simpleType or a complexType with simple or mixed content required')
                raise XMLSchemaValueError(msg % self.base_type)
        result: Any
        for result in base_type.iter_encode(obj, validation):
            if isinstance(result, XMLSchemaValidationError):
                yield result
                if isinstance(result, XMLSchemaEncodeError):
                    yield (str(obj) if validation == 'skip' else None)
                    return
            else:
                if self.validators and obj is not None:
                    if isinstance(obj, (str, bytes)) and self.primitive_type.to_python is not str and isinstance(obj, self.primitive_type.instance_types):
                        try:
                            obj = self.primitive_type.to_python(obj)
                        except (ValueError, DecimalException, TypeError):
                            pass
                    for validator in self.validators:
                        try:
                            validator(obj)
                        except XMLSchemaValidationError as err:
                            yield err
                if self.patterns:
                    if not isinstance(self.primitive_type, XsdUnion):
                        try:
                            self.patterns(result)
                        except XMLSchemaValidationError as err:
                            yield err
                    elif 'patterns' not in kwargs:
                        kwargs['patterns'] = self.patterns
                yield result
                return

    def is_list(self) -> bool:
        return self.primitive_type.is_list()

    def is_union(self) -> bool:
        return self.primitive_type.is_union()