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
class ConfigurationSetInputEndpoint(WindowsAzureData):

    def __init__(self, name='', protocol='', port='', local_port='', load_balanced_endpoint_set_name='', enable_direct_server_return=False):
        self.enable_direct_server_return = enable_direct_server_return
        self.load_balanced_endpoint_set_name = load_balanced_endpoint_set_name
        self.local_port = local_port
        self.name = name
        self.port = port
        self.load_balancer_probe = LoadBalancerProbe()
        self.protocol = protocol