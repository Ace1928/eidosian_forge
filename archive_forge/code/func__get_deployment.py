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
def _get_deployment(self, **kwargs):
    _service_name = kwargs['service_name']
    _deployment_slot = kwargs['deployment_slot']
    response = self._perform_get(self._get_deployment_path_using_slot(_service_name, _deployment_slot), None)
    self.raise_for_response(response, 200)
    return self._parse_response(response, Deployment)