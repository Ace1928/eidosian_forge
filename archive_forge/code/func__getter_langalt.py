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
def _getter_langalt(namespace: str, name: str) -> Callable[['XmpInformation'], Optional[Dict[Any, Any]]]:

    def get(self: 'XmpInformation') -> Optional[Dict[Any, Any]]:
        cached = self.cache.get(namespace, {}).get(name)
        if cached:
            return cached
        retval = {}
        for element in self.get_element('', namespace, name):
            alts = element.getElementsByTagNameNS(RDF_NAMESPACE, 'Alt')
            if len(alts):
                for alt in alts:
                    for item in alt.getElementsByTagNameNS(RDF_NAMESPACE, 'li'):
                        value = self._get_text(item)
                        retval[item.getAttribute('xml:lang')] = value
            else:
                retval['x-default'] = self._get_text(element)
        ns_cache = self.cache.setdefault(namespace, {})
        ns_cache[name] = retval
        return retval
    return get