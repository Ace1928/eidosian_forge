import io
import xml.dom
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE, domreg
from xml.dom.minicompat import *
from xml.dom.xmlbuilder import DOMImplementationLS, DocumentLS
def _nssplit(qualifiedName):
    fields = qualifiedName.split(':', 1)
    if len(fields) == 2:
        return fields
    else:
        return (None, fields[0])