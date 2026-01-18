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
def _parse_response_body_from_xml_text(self, response, return_type):
    """
        parse the xml and fill all the data into a class of return_type
        """
    respbody = response.body
    doc = minidom.parseString(respbody)
    return_obj = return_type()
    for node in self._get_child_nodes(doc, return_type.__name__):
        self._fill_data_to_return_object(node, return_obj)
    return_obj.status = response.status
    return return_obj