from xml.sax._exceptions import *
from xml.sax.handler import feature_validation, feature_namespaces
from xml.sax.handler import feature_namespace_prefixes
from xml.sax.handler import feature_external_ges, feature_external_pes
from xml.sax.handler import feature_string_interning
from xml.sax.handler import property_xml_string, property_interning_dict
import sys
from xml.sax import xmlreader, saxutils, handler
def _reset_lex_handler_prop(self):
    lex = self._lex_handler_prop
    parser = self._parser
    if lex is None:
        parser.CommentHandler = None
        parser.StartCdataSectionHandler = None
        parser.EndCdataSectionHandler = None
        parser.StartDoctypeDeclHandler = None
        parser.EndDoctypeDeclHandler = None
    else:
        parser.CommentHandler = lex.comment
        parser.StartCdataSectionHandler = lex.startCDATA
        parser.EndCdataSectionHandler = lex.endCDATA
        parser.StartDoctypeDeclHandler = self.start_doctype_decl
        parser.EndDoctypeDeclHandler = lex.endDTD