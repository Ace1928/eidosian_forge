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
def _fill_data_minidom(self, xmldoc, element_name, data_member):
    xmlelements = self._get_child_nodes(xmldoc, self._get_serialization_name(element_name))
    if not xmlelements or not xmlelements[0].childNodes:
        return None
    value = xmlelements[0].firstChild.nodeValue
    if data_member is None:
        return value
    elif isinstance(data_member, datetime):
        return self._to_datetime(value)
    elif type(data_member) is bool:
        return value.lower() != 'false'
    elif type(data_member) is str:
        return _real_unicode(value)
    else:
        return type(data_member)(value)