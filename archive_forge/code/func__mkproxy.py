from xml.sax._exceptions import *
from xml.sax.handler import feature_validation, feature_namespaces
from xml.sax.handler import feature_namespace_prefixes
from xml.sax.handler import feature_external_ges, feature_external_pes
from xml.sax.handler import feature_string_interning
from xml.sax.handler import property_xml_string, property_interning_dict
import sys
from xml.sax import xmlreader, saxutils, handler
def _mkproxy(o):
    return o