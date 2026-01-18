from __future__ import print_function, absolute_import
import threading
import warnings
from lxml import etree as _etree
from .common import DTDForbidden, EntitiesForbidden, NotSupportedError
class GlobalParserTLS(threading.local):
    """Thread local context for custom parser instances"""
    parser_config = {'resolve_entities': False}
    element_class = RestrictedElement

    def createDefaultParser(self):
        parser = _etree.XMLParser(**self.parser_config)
        element_class = self.element_class
        if self.element_class is not None:
            lookup = _etree.ElementDefaultClassLookup(element=element_class)
            parser.set_element_class_lookup(lookup)
        return parser

    def setDefaultParser(self, parser):
        self._default_parser = parser

    def getDefaultParser(self):
        parser = getattr(self, '_default_parser', None)
        if parser is None:
            parser = self.createDefaultParser()
            self.setDefaultParser(parser)
        return parser