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
def _ex_complete_async_azure_operation(self, response=None, operation_type='create_node'):
    request_id = self._parse_response_for_async_op(response)
    operation_status = self._get_operation_status(request_id.request_id)
    timeout = 60 * 5
    waittime = 0
    interval = 5
    while operation_status.status == 'InProgress' and waittime < timeout:
        operation_status = self._get_operation_status(request_id)
        if operation_status.status == 'Succeeded':
            break
        waittime += interval
        time.sleep(interval)
    if operation_status.status == 'Failed':
        raise LibcloudError('Message: Async request for operation %s has failed' % operation_type, driver=self.connection.driver)