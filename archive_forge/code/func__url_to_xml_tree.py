from __future__ import unicode_literals
import sys
import copy
import hashlib
import logging
import os
import tempfile
import warnings
from . import __author__, __copyright__, __license__, __version__, TIMEOUT
from .simplexml import SimpleXMLElement, TYPE_MAP, REVERSE_TYPE_MAP, Struct
from .transport import get_http_wrapper, set_http_wrapper, get_Http
from .helpers import Alias, fetch, sort_dict, make_key, process_element, \
from .wsse import UsernameToken
def _url_to_xml_tree(self, url, cache, force_download):
    """Unmarshall the WSDL at the given url into a tree of SimpleXMLElement nodes"""
    xml = fetch(url, self.http, cache, force_download, self.wsdl_basedir, self.http_headers)
    wsdl = SimpleXMLElement(xml, namespace=self.wsdl_uri)
    self.namespace = ''
    self.documentation = unicode(wsdl('documentation', error=False)) or ''
    imported_wsdls = {}
    for element in wsdl.children() or []:
        if element.get_local_name() in 'import':
            wsdl_namespace = element['namespace']
            wsdl_location = element['location']
            if wsdl_location is None:
                log.warning('WSDL location not provided for %s!' % wsdl_namespace)
                continue
            if wsdl_location in imported_wsdls:
                log.warning('WSDL %s already imported!' % wsdl_location)
                continue
            imported_wsdls[wsdl_location] = wsdl_namespace
            log.debug('Importing wsdl %s from %s' % (wsdl_namespace, wsdl_location))
            xml = fetch(wsdl_location, self.http, cache, force_download, self.wsdl_basedir, self.http_headers)
            imported_wsdl = SimpleXMLElement(xml, namespace=self.xsd_uri)
            wsdl.import_node(imported_wsdl)
    return wsdl