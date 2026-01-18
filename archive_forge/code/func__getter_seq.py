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
def _getter_seq(namespace: str, name: str, converter: Callable[[Any], Any]=_identity) -> Callable[['XmpInformation'], Optional[List[Any]]]:

    def get(self: 'XmpInformation') -> Optional[List[Any]]:
        cached = self.cache.get(namespace, {}).get(name)
        if cached:
            return cached
        retval = []
        for element in self.get_element('', namespace, name):
            seqs = element.getElementsByTagNameNS(RDF_NAMESPACE, 'Seq')
            if len(seqs):
                for seq in seqs:
                    for item in seq.getElementsByTagNameNS(RDF_NAMESPACE, 'li'):
                        value = self._get_text(item)
                        value = converter(value)
                        retval.append(value)
            else:
                value = converter(self._get_text(element))
                retval.append(value)
        ns_cache = self.cache.setdefault(namespace, {})
        ns_cache[name] = retval
        return retval
    return get