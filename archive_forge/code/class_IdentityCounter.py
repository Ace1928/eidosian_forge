import re
import math
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Pattern, \
from elementpath import XPath2Parser, ElementPathError, XPathToken, XPathContext, \
from ..exceptions import XMLSchemaTypeError, XMLSchemaValueError
from ..names import XSD_QNAME, XSD_UNIQUE, XSD_KEY, XSD_KEYREF, XSD_SELECTOR, XSD_FIELD
from ..translation import gettext as _
from ..helpers import get_qname, get_extended_qname
from ..aliases import ElementType, SchemaType, NamespacesType, AtomicValueType
from .exceptions import XMLSchemaNotBuiltError
from .xsdbase import XsdComponent
from .attributes import XsdAttribute
from .wildcards import XsdAnyElement
from . import elements
class IdentityCounter:

    def __init__(self, identity: XsdIdentity, elem: ElementType) -> None:
        self.counter: Counter[IdentityCounterType] = Counter[IdentityCounterType]()
        self.identity = identity
        self.elem = elem
        self.enabled = True
        self.elements = None

    def __repr__(self) -> str:
        return '%s%r' % (self.__class__.__name__[:-7], self.counter)

    def reset(self, elem: ElementType) -> None:
        self.counter.clear()
        self.elem = elem
        self.enabled = True
        self.elements = None

    def increase(self, fields: IdentityCounterType) -> None:
        self.counter[fields] += 1
        if self.counter[fields] == 2:
            msg = _('duplicated value {0!r} for {1!r}')
            raise XMLSchemaValueError(msg.format(fields, self.identity))