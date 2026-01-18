import warnings
from copy import copy as _copy
from decimal import Decimal
from types import GeneratorType
from typing import TYPE_CHECKING, cast, Any, Dict, Iterator, List, Optional, \
from xml.etree.ElementTree import Element
from elementpath import XPath2Parser, ElementPathError, XPathContext, XPathToken, \
from elementpath.datatypes import AbstractDateTime, Duration, AbstractBinary
from ..exceptions import XMLSchemaTypeError, XMLSchemaValueError
from ..names import XSD_COMPLEX_TYPE, XSD_SIMPLE_TYPE, XSD_ALTERNATIVE, \
from ..aliases import ElementType, SchemaType, BaseXsdType, SchemaElementType, \
from ..translation import gettext as _
from ..helpers import get_qname, get_namespace, etree_iter_location_hints, \
from .. import dataobjects
from ..converters import ElementData, XMLSchemaConverter
from ..xpath import XsdSchemaProtocol, XsdElementProtocol, XMLSchemaProxy, \
from ..resources import XMLResource
from .exceptions import XMLSchemaNotBuiltError, XMLSchemaValidationError, \
from .helpers import get_xsd_derivation_attribute
from .xsdbase import XSD_TYPE_DERIVATIONS, XSD_ELEMENT_DERIVATIONS, \
from .particles import ParticleMixin, OccursCalculator
from .identities import XsdIdentity, XsdKey, XsdUnique, \
from .simple_types import XsdSimpleType
from .attributes import XsdAttribute
from .wildcards import XsdAnyElement
def _parse_alternatives(self) -> None:
    alternatives = []
    has_test = True
    for child in self.elem:
        if child.tag == XSD_ALTERNATIVE:
            alternatives.append(XsdAlternative(child, self.schema, self))
            if not has_test:
                msg = _('test attribute missing in non-final alternative')
                self.parse_error(msg)
            has_test = 'test' in child.attrib
    if alternatives:
        self.alternatives = alternatives