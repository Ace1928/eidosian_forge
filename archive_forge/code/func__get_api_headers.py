from __future__ import absolute_import, division, print_function
import os
import sys
import copy
import json
import logging
import time
from datetime import datetime, timedelta
from ssl import SSLError
def _get_api_headers(self, tenant, tenant_uuid, timeout, headers, api_version):
    """
        returns the headers that are passed to the requests.Session api calls.
        """
    api_hdrs = copy.deepcopy(self.headers)
    api_hdrs.update({'Referer': self.prefix, 'Content-Type': 'application/json'})
    api_hdrs['timeout'] = str(timeout)
    if self.key in sessionDict and 'csrftoken' in sessionDict.get(self.key):
        api_hdrs['X-CSRFToken'] = sessionDict.get(self.key)['csrftoken']
    else:
        self.authenticate_session()
        api_hdrs['X-CSRFToken'] = sessionDict.get(self.key)['csrftoken']
    if api_version:
        api_hdrs['X-Avi-Version'] = api_version
    elif self.avi_credentials.api_version:
        api_hdrs['X-Avi-Version'] = self.avi_credentials.api_version
    if tenant:
        tenant_uuid = None
    elif tenant_uuid:
        tenant = None
    else:
        tenant = self.avi_credentials.tenant
        tenant_uuid = self.avi_credentials.tenant_uuid
    if tenant_uuid:
        api_hdrs.update({'X-Avi-Tenant-UUID': '%s' % tenant_uuid})
        api_hdrs.pop('X-Avi-Tenant', None)
    elif tenant:
        api_hdrs.update({'X-Avi-Tenant': '%s' % tenant})
        api_hdrs.pop('X-Avi-Tenant-UUID', None)
    if self.user_hdrs:
        api_hdrs.update(self.user_hdrs)
    if headers:
        api_hdrs.update(headers)
    return api_hdrs