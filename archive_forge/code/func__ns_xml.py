import datetime
import logging
from lxml import etree
import io
import warnings
import prov
import prov.identifier
from prov.model import DEFAULT_NAMESPACES, sorted_attributes
from prov.constants import *  # NOQA
from prov.serializers import Serializer
def _ns_xml(tag):
    NS_XML = 'http://www.w3.org/XML/1998/namespace'
    return _ns(NS_XML, tag)