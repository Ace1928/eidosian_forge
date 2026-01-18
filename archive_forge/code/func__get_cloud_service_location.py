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
def _get_cloud_service_location(self, service_name=None):
    if not service_name:
        raise ValueError('service_name is required.')
    res = self._perform_get('%s?embed-detail=False' % self._get_hosted_service_path(service_name), HostedService)
    _affinity_group = res.hosted_service_properties.affinity_group
    _cloud_service_location = res.hosted_service_properties.location
    if _affinity_group is not None and _affinity_group != '':
        return self.service_location(True, _affinity_group)
    elif _cloud_service_location is not None:
        return self.service_location(False, _cloud_service_location)
    else:
        return None