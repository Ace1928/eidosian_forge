import datetime
import decimal
import re
from typing import (
from xml.dom.minidom import Document, parseString
from xml.dom.minidom import Element as XmlElement
from xml.parsers.expat import ExpatError
from ._utils import StreamType, deprecate_no_replacement
from .errors import PdfReadError
from .generic import ContentStream, PdfObject
def _getter_bag(namespace: str, name: str) -> Callable[['XmpInformation'], Optional[List[str]]]:

    def get(self: 'XmpInformation') -> Optional[List[str]]:
        cached = self.cache.get(namespace, {}).get(name)
        if cached:
            return cached
        retval = []
        for element in self.get_element('', namespace, name):
            bags = element.getElementsByTagNameNS(RDF_NAMESPACE, 'Bag')
            if len(bags):
                for bag in bags:
                    for item in bag.getElementsByTagNameNS(RDF_NAMESPACE, 'li'):
                        value = self._get_text(item)
                        retval.append(value)
        ns_cache = self.cache.setdefault(namespace, {})
        ns_cache[name] = retval
        return retval
    return get