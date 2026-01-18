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
class OSVirtualHardDisk(WindowsAzureData):

    def __init__(self, source_image_name=None, media_link=None, host_caching=None, disk_label=None, disk_name=None):
        self.source_image_name = source_image_name
        self.media_link = media_link
        self.host_caching = host_caching
        self.disk_label = disk_label
        self.disk_name = disk_name
        self.os = ''