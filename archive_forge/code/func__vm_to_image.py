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
def _vm_to_image(self, data):
    return NodeImage(id=data.name, name=data.label, driver=self.connection.driver, extra={'os': data.os_disk_configuration.os, 'category': data.category, 'location': data.location, 'media_link': data.os_disk_configuration.media_link, 'affinity_group': data.affinity_group, 'deployment_name': data.deployment_name, 'vm_image': True})