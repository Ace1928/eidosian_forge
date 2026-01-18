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
class KeyrefCounter(IdentityCounter):
    identity: XsdKeyref

    def increase(self, fields: IdentityCounterType) -> None:
        self.counter[fields] += 1

    def iter_errors(self, identities: IdentityMapType) -> Iterator[XMLSchemaValueError]:
        refer_values = identities[self.identity.refer].counter
        for v in filter(lambda x: x not in refer_values, self.counter):
            if len(v) == 1 and v[0] in refer_values:
                continue
            elif self.counter[v] > 1:
                msg = 'value {} not found for {!r} ({} times)'
                yield XMLSchemaValueError(msg.format(v, self.identity.refer, self.counter[v]))
            else:
                msg = 'value {} not found for {!r}'
                yield XMLSchemaValueError(msg.format(v, self.identity.refer))