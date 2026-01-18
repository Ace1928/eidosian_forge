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
class VMImage(WindowsAzureData):

    def __init__(self):
        self.name = ''
        self.label = ''
        self.category = ''
        self.os_disk_configuration = OSDiskConfiguration()
        self.service_name = ''
        self.deployment_name = ''
        self.role_name = ''
        self.location = ''
        self.affinity_group = ''