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
def _create_storage_account(self, **kwargs):
    if kwargs['is_affinity_group'] is True:
        response = self._perform_post(self._get_storage_service_path(), AzureXmlSerializer.create_storage_service_input_to_xml(kwargs['service_name'], kwargs['service_name'], self._encode_base64(kwargs['service_name']), kwargs['location'], None, True, None))
        self.raise_for_response(response, 202)
    else:
        response = self._perform_post(self._get_storage_service_path(), AzureXmlSerializer.create_storage_service_input_to_xml(kwargs['service_name'], kwargs['service_name'], self._encode_base64(kwargs['service_name']), None, kwargs['location'], True, None))
        self.raise_for_response(response, 202)
    self._ex_complete_async_azure_operation(response, 'create_storage_account')