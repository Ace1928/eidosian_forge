from typing import cast, Any, Callable, Dict, Iterable, Iterator, List, Optional, \
from elementpath import SchemaElementNode, build_schema_node_tree
from ..exceptions import XMLSchemaValueError
from ..names import XSI_NAMESPACE, XSD_ANY, XSD_ANY_ATTRIBUTE, \
from ..aliases import ElementType, SchemaType, SchemaElementType, SchemaAttributeType, \
from ..translation import gettext as _
from ..helpers import get_namespace, raw_xml_encode
from ..xpath import XsdSchemaProtocol, XsdElementProtocol, XMLSchemaProxy, ElementPathMixin
from .xsdbase import ValidationMixin, XsdComponent
from .particles import ParticleMixin
from . import elements
class XsdAnyAttribute(XsdWildcard, ValidationMixin[Tuple[str, str], DecodedValueType]):
    """
    Class for XSD 1.0 *anyAttribute* wildcards.

    ..  <anyAttribute
          id = ID
          namespace = ((##any | ##other) | List of (anyURI | (##targetNamespace | ##local)) )
          processContents = (lax | skip | strict) : strict
          {any attributes with non-schema namespace . . .}>
          Content: (annotation?)
        </anyAttribute>
    """
    copy: Callable[['XsdAnyAttribute'], 'XsdAnyAttribute']
    _ADMITTED_TAGS = {XSD_ANY_ATTRIBUTE}
    use = None
    inheritable = False

    def match(self, name: Optional[str], default_namespace: Optional[str]=None, resolve: bool=False, **kwargs: Any) -> Optional[SchemaAttributeType]:
        """
        Returns the attribute wildcard if name is matching the name provided
        as argument, `None` otherwise.

        :param name: a local or fully-qualified name.
        :param default_namespace: used when it's not `None` and not empty for         completing local name arguments.
        :param resolve: when `True` it doesn't return the wildcard but try to         resolve and return the attribute matching the name.
        :param kwargs: additional options that can be used by certain components.
        """
        if not name or not self.is_matching(name, default_namespace, **kwargs):
            return None
        elif not resolve:
            return self
        try:
            if name[0] != '{' and default_namespace:
                return self.maps.lookup_attribute(f'{{{default_namespace}}}{name}')
            else:
                return self.maps.lookup_attribute(name)
        except LookupError:
            return None

    def iter_decode(self, obj: Tuple[str, str], validation: str='lax', **kwargs: Any) -> IterDecodeType[DecodedValueType]:
        name, value = obj
        if not self.is_matching(name):
            reason = _('attribute %r not allowed') % name
            yield self.validation_error(validation, reason, obj, **kwargs)
        if self.process_contents == 'skip':
            if 'process_skipped' not in kwargs or not kwargs['process_skipped']:
                return
        if self.maps.load_namespace(get_namespace(name)):
            try:
                xsd_attribute = self.maps.lookup_attribute(name)
            except LookupError:
                if validation != 'skip' and self.process_contents == 'strict':
                    reason = _('attribute %r not found') % name
                    yield self.validation_error(validation, reason, obj, **kwargs)
            else:
                yield from xsd_attribute.iter_decode(value, validation, **kwargs)
                return
        elif validation != 'skip' and self.process_contents == 'strict':
            reason = _('unavailable namespace {!r}').format(get_namespace(name))
            yield self.validation_error(validation, reason, **kwargs)
        yield value

    def iter_encode(self, obj: Tuple[str, AtomicValueType], validation: str='lax', **kwargs: Any) -> IterEncodeType[EncodedValueType]:
        name, value = obj
        namespace = get_namespace(name)
        if not self.is_namespace_allowed(namespace):
            reason = _('attribute %r not allowed') % name
            yield self.validation_error(validation, reason, obj, **kwargs)
        if self.process_contents == 'skip':
            if 'process_skipped' not in kwargs or not kwargs['process_skipped']:
                return
        if self.maps.load_namespace(namespace):
            try:
                xsd_attribute = self.maps.lookup_attribute(name)
            except LookupError:
                if validation != 'skip' and self.process_contents == 'strict':
                    reason = _('attribute %r not found') % name
                    yield self.validation_error(validation, reason, obj, **kwargs)
            else:
                yield from xsd_attribute.iter_encode(value, validation, **kwargs)
                return
        elif validation != 'skip' and self.process_contents == 'strict':
            reason = _('unavailable namespace {!r}').format(get_namespace(name))
            yield self.validation_error(validation, reason, **kwargs)
        yield raw_xml_encode(value)