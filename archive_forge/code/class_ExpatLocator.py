from xml.sax._exceptions import *
from xml.sax.handler import feature_validation, feature_namespaces
from xml.sax.handler import feature_namespace_prefixes
from xml.sax.handler import feature_external_ges, feature_external_pes
from xml.sax.handler import feature_string_interning
from xml.sax.handler import property_xml_string, property_interning_dict
import sys
from xml.sax import xmlreader, saxutils, handler
class ExpatLocator(xmlreader.Locator):
    """Locator for use with the ExpatParser class.

    This uses a weak reference to the parser object to avoid creating
    a circular reference between the parser and the content handler.
    """

    def __init__(self, parser):
        self._ref = _mkproxy(parser)

    def getColumnNumber(self):
        parser = self._ref
        if parser._parser is None:
            return None
        return parser._parser.ErrorColumnNumber

    def getLineNumber(self):
        parser = self._ref
        if parser._parser is None:
            return 1
        return parser._parser.ErrorLineNumber

    def getPublicId(self):
        parser = self._ref
        if parser is None:
            return None
        return parser._source.getPublicId()

    def getSystemId(self):
        parser = self._ref
        if parser is None:
            return None
        return parser._source.getSystemId()