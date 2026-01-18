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
def _get_serialization_name(self, element_name):
    """
        Converts a Python name into a serializable name.
        """
    known = _KNOWN_SERIALIZATION_XFORMS.get(element_name)
    if known is not None:
        return known
    if element_name.startswith('x_ms_'):
        return element_name.replace('_', '-')
    if element_name.endswith('_id'):
        element_name = element_name.replace('_id', 'ID')
    for name in ['content_', 'last_modified', 'if_', 'cache_control']:
        if element_name.startswith(name):
            element_name = element_name.replace('_', '-_')
    return ''.join((name.capitalize() for name in element_name.split('_')))