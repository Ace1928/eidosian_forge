import re
import copy
import time
import base64
import random
import collections
from xml.dom import minidom
from datetime import datetime
from xml.sax.saxutils import escape as xml_escape
from libcloud.utils.py3 import ET, httplib, urlparse
from libcloud.utils.py3 import urlquote as url_quote
from libcloud.utils.py3 import _real_unicode, ensure_string
from libcloud.utils.misc import ReprMixin
from libcloud.common.azure import AzureRedirectException, AzureServiceManagementConnection
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.compute.types import NodeState
from libcloud.compute.providers import Provider
def _fill_scalar_list_of(self, xmldoc, element_type, parent_xml_element_name, xml_element_name):
    xmlelements = self._get_child_nodes(xmldoc, parent_xml_element_name)
    if xmlelements:
        xmlelements = self._get_child_nodes(xmlelements[0], xml_element_name)
        return [self._get_node_value(xmlelement, element_type) for xmlelement in xmlelements]