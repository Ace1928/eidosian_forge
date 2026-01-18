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
def _to_node_size(self, data):
    """
        Convert the AZURE_COMPUTE_INSTANCE_TYPES into NodeSize
        """
    return NodeSize(id=data['id'], name=data['name'], ram=data['ram'], disk=data['disk'], bandwidth=data['bandwidth'], price=data['price'], driver=self.connection.driver, extra={'max_data_disks': data['max_data_disks'], 'cores': data['cores']})